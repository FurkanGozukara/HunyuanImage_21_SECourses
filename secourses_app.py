import os
import sys
import warnings
import subprocess
import json
import platform
import webbrowser
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime
import argparse
import random
import torch

warnings.filterwarnings("ignore")

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import gradio as gr
    from PIL import Image
    from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline
    from huggingface_hub import snapshot_download
    import modelscope
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install with: pip install -r requirements_gradio.txt")
    print("For checkpoint downloads, also install: pip install -U 'huggingface_hub[cli]' modelscope")
    sys.exit(1)


# Configuration
BASE_DIR = Path('./ckpts')
OUTPUTS_DIR = Path('./outputs')
CONFIGS_DIR = Path('./configs')
LAST_CONFIG_FILE = CONFIGS_DIR / '_last_config.json'

# Create necessary directories
BASE_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)


class ImageSaver:
    """Handles saving images with unique names and metadata."""
    
    def __init__(self, output_dir: Path = OUTPUTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def get_next_filename(self) -> Tuple[str, str]:
        """Get next available filename with format 0001.png."""
        existing_files = list(self.output_dir.glob('*.png'))
        existing_numbers = []
        
        for file in existing_files:
            try:
                # Extract number from filename like 0001.png or 0001_pre_refiner.png
                base_name = file.stem.split('_')[0]
                existing_numbers.append(int(base_name))
            except (ValueError, IndexError):
                continue
                
        next_number = max(existing_numbers, default=0) + 1
        filename = f"{next_number:04d}.png"
        return filename, f"{next_number:04d}"
        
    def save_image(self, image: Image.Image, metadata: Dict, is_pre_refiner: bool = False) -> str:
        """Save image with metadata."""
        base_name, number = self.get_next_filename()
        
        if is_pre_refiner:
            filename = f"{number}_pre_refiner.png"
        else:
            filename = base_name
            
        filepath = self.output_dir / filename
        
        # Save image
        image.save(filepath)
        
        # Save metadata
        metadata_path = filepath.with_suffix('.txt')
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['filename'] = filename
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return str(filepath)


class ConfigManager:
    """Manages saving and loading configurations."""
    
    def __init__(self, configs_dir: Path = CONFIGS_DIR):
        self.configs_dir = configs_dir
        self.configs_dir.mkdir(exist_ok=True)
        
    def save_config(self, config_name: str, params: Dict) -> bool:
        """Save configuration to file."""
        try:
            config_path = self.configs_dir / f"{config_name}.json"
            with open(config_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            # Save as last used config
            with open(LAST_CONFIG_FILE, 'w') as f:
                json.dump(params, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
            
    def load_config(self, config_name: str) -> Optional[Dict]:
        """Load configuration from file."""
        try:
            config_path = self.configs_dir / f"{config_name}.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    params = json.load(f)
                    
                # Save as last used config
                with open(LAST_CONFIG_FILE, 'w') as f:
                    json.dump(params, f, indent=2)
                return params
        except Exception as e:
            print(f"Error loading config: {e}")
        return None
        
    def load_last_config(self) -> Optional[Dict]:
        """Load last used configuration."""
        try:
            if LAST_CONFIG_FILE.exists():
                with open(LAST_CONFIG_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading last config: {e}")
        return None
        
    def list_configs(self) -> List[str]:
        """List available configurations."""
        configs = []
        for config_file in self.configs_dir.glob('*.json'):
            if not config_file.name.startswith('_'):
                configs.append(config_file.stem)
        return sorted(configs)


class CheckpointDownloader:
    """Handles downloading of all required checkpoints for HunyuanImage."""
    
    def __init__(self, base_dir: Path = BASE_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)
        print(f'Downloading checkpoints to: {self.base_dir}')
        
        # Define all required checkpoints with corrected paths
        self.checkpoints = {
            "main_model": {
                "repo_id": "tencent/HunyuanImage-2.1",
                "local_dir": self.base_dir,
            },
            "mllm_encoder": {
                "repo_id": "Qwen/Qwen2.5-VL-7B-Instruct", 
                "local_dir": self.base_dir / "text_encoder" / "llm",
            },
            "byt5_encoder": {
                "repo_id": "google/byt5-small",
                "local_dir": self.base_dir / "text_encoder" / "byt5-small", 
            },
            "glyph_encoder": {
                "repo_id": "AI-ModelScope/Glyph-SDXL-v2",
                "local_dir": self.base_dir / "text_encoder" / "Glyph-SDXL-v2",
                "use_modelscope": True
            }
        }
    
    def download_checkpoint(self, checkpoint_name: str, progress_callback=None) -> Tuple[bool, str]:
        """Download a specific checkpoint."""
        if checkpoint_name not in self.checkpoints:
            return False, f"Unknown checkpoint: {checkpoint_name}"
        
        config = self.checkpoints[checkpoint_name]
        local_dir = config["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if config.get("use_modelscope", False):
                return self._download_with_modelscope(config, progress_callback)
            else:
                return self._download_with_hf(config, progress_callback)
        except Exception as e:
            return False, f"Download failed: {str(e)}"
    
    def _download_with_hf(self, config: Dict, progress_callback=None) -> Tuple[bool, str]:
        """Download using huggingface_hub."""
        repo_id = config["repo_id"]
        local_dir = config["local_dir"]
        
        if progress_callback:
            progress_callback(f"Downloading {repo_id}...")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            return True, f"Successfully downloaded {repo_id}"
        except Exception as e:
            return False, f"HF download failed: {str(e)}"
    
    def _download_with_modelscope(self, config: Dict, progress_callback=None) -> Tuple[bool, str]:
        """Download using modelscope."""
        repo_id = config["repo_id"]
        local_dir = config["local_dir"]
        
        if progress_callback:
            progress_callback(f"Downloading {repo_id} via ModelScope...")
        print(f"Downloading {repo_id} via ModelScope...")
        
        try:
            cmd = [
                "modelscope", "download", 
                "--model", repo_id,
                "--local_dir", str(local_dir)
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True, f"Successfully downloaded {repo_id} via ModelScope"
        except subprocess.CalledProcessError as e:
            return False, f"ModelScope download failed: {e.stderr}"
        except FileNotFoundError:
            return False, "ModelScope CLI not found. Install with: pip install modelscope"
    
    def download_all_checkpoints(self, progress_callback=None) -> Tuple[bool, str, Dict[str, any]]:
        """Download all checkpoints."""
        results = {}
        for name, _ in self.checkpoints.items():
            if progress_callback:
                progress_callback(f"Starting download of {name}...")
            
            success, message = self.download_checkpoint(name, progress_callback)
            results[name] = {"success": success, "message": message}
            
            if not success:
                return False, f"Failed to download {name}: {message}", results
        return True, "All checkpoints downloaded successfully", results


def load_pipeline(use_distilled: bool = False, 
                 device: str = "cuda",
                 enable_dit_offloading: bool = True,
                 enable_reprompt_offloading: bool = True,
                 enable_refiner_offloading: bool = True):
    """Load the HunyuanImage pipeline with configurable offloading."""
    try:
        print(f"Loading HunyuanImage pipeline (distilled={use_distilled})...")
        print(f"  DiT offloading: {enable_dit_offloading}")
        print(f"  Reprompt offloading: {enable_reprompt_offloading}")
        print(f"  Refiner offloading: {enable_refiner_offloading}")
        
        # Set model root to our models directory
        os.environ['HUNYUANIMAGE_V2_1_MODEL_ROOT'] = str(BASE_DIR)
        
        model_name = "hunyuanimage-v2.1-distilled" if use_distilled else "hunyuanimage-v2.1"
        pipeline = HunyuanImagePipeline.from_pretrained(
            model_name=model_name,
            device=device,
            enable_dit_offloading=enable_dit_offloading,
            enable_reprompt_model_offloading=enable_reprompt_offloading,
            enable_refiner_offloading=enable_refiner_offloading
        )
        pipeline.to('cpu')
        
        # Don't set reprompt flag here - it will be set dynamically
        
        # Don't setup refiner here - it will be loaded on demand when needed
        # This saves memory if refiner is never used
        
        print("✓ Pipeline loaded successfully")
        return pipeline
    except Exception as e:
        error_msg = f"Error loading pipeline: {str(e)}"
        print(f"✗ {error_msg}")
        raise


# Global pipeline variable - will be loaded on demand
pipeline = None
current_model_type = None
current_offloading_settings = {
    'dit': True,
    'reprompt': True,
    'refiner': True
}
current_reprompt_loaded = False


class HunyuanImageApp:
    def __init__(self, auto_load: bool = True, use_distilled: bool = False, device: str = "cuda"):
        """Initialize the HunyuanImage Gradio app."""
        global pipeline, current_model_type, current_offloading_settings
        self.device = device
        self.image_saver = ImageSaver()
        self.config_manager = ConfigManager()
        self.initial_model_type = "distilled" if use_distilled else "regular"
        
        # Don't auto-load pipeline anymore - will load on first generation
        # This allows user to configure VRAM settings before loading
        self.pipeline = pipeline

    def print_peak_memory(self):
        """Print peak GPU memory usage."""
        stats = torch.cuda.memory_stats()
        peak_bytes_requirement = stats["allocated_bytes.all.peak"]
        print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")

    def switch_model(self, model_type: str, enable_dit_offloading: bool, 
                    enable_reprompt_offloading: bool, enable_refiner_offloading: bool,
                    use_reprompt: bool, auto_enhance: bool) -> str:
        """Switch between models and update offloading settings."""
        global pipeline, current_model_type, current_offloading_settings, current_reprompt_loaded
        
        try:
            use_distilled = (model_type == "distilled")
            
            # Only check for DiT offloading changes that require pipeline reload
            # Reprompt and Refiner changes do NOT require pipeline reload
            dit_offloading_changed = current_offloading_settings['dit'] != enable_dit_offloading
            
            # Check if we need to reload the entire pipeline (only for model type or DiT offloading)
            if current_model_type != model_type or dit_offloading_changed:
                print(f"Reloading pipeline with new settings...")
                print(f"  Model: {model_type}")
                print(f"  DiT offloading: {enable_dit_offloading}")
                
                # Clear GPU memory
                if pipeline is not None:
                    del pipeline
                    torch.cuda.empty_cache()
                
                # Load new pipeline with updated settings
                pipeline = load_pipeline(
                    use_distilled=use_distilled, 
                    device=self.device,
                    enable_dit_offloading=enable_dit_offloading,
                    enable_reprompt_offloading=enable_reprompt_offloading,
                    enable_refiner_offloading=enable_refiner_offloading
                )
                self.pipeline = pipeline
                current_model_type = model_type
                current_offloading_settings = {
                    'dit': enable_dit_offloading,
                    'reprompt': enable_reprompt_offloading,
                    'refiner': enable_refiner_offloading
                }
                current_reprompt_loaded = False  # Reset reprompt state after pipeline reload
                
                return f"Pipeline reloaded with updated settings"
            else:
                # Pipeline doesn't need reload, just update offloading settings if needed
                if current_offloading_settings['reprompt'] != enable_reprompt_offloading:
                    current_offloading_settings['reprompt'] = enable_reprompt_offloading
                    if pipeline is not None:
                        pipeline.enable_reprompt_model_offloading = enable_reprompt_offloading
                        
                if current_offloading_settings['refiner'] != enable_refiner_offloading:
                    current_offloading_settings['refiner'] = enable_refiner_offloading
                    if pipeline is not None:
                        pipeline.enable_refiner_offloading = enable_refiner_offloading
                        
                return f"Settings updated without pipeline reload"
                
        except Exception as e:
            return f"Error updating pipeline: {str(e)}"
    
    def ensure_pipeline_loaded(self, model_type: str, enable_dit_offloading: bool, 
                             enable_reprompt_offloading: bool, enable_refiner_offloading: bool,
                             use_reprompt: bool, auto_enhance: bool = False):
        """Ensure pipeline is loaded with the correct settings."""
        global pipeline, current_model_type, current_offloading_settings, current_reprompt_loaded
        
        if pipeline is None:
            # First time loading - use the provided settings
            print("Loading pipeline for the first time with user-selected VRAM optimizations...")
            use_distilled = (model_type == "distilled")
            pipeline = load_pipeline(
                use_distilled=use_distilled,
                device=self.device,
                enable_dit_offloading=enable_dit_offloading,
                enable_reprompt_offloading=enable_reprompt_offloading,
                enable_refiner_offloading=enable_refiner_offloading
            )
            self.pipeline = pipeline
            current_model_type = model_type
            current_offloading_settings = {
                'dit': enable_dit_offloading,
                'reprompt': enable_reprompt_offloading,
                'refiner': enable_refiner_offloading
            }
            current_reprompt_loaded = False
            print(f"✓ Pipeline loaded with user settings")
        
        # Handle reprompt model loading separately (doesn't require pipeline reload)
        self._ensure_reprompt_loaded(use_reprompt or auto_enhance)
    
    def _ensure_reprompt_loaded(self, should_load: bool):
        """Ensure reprompt model is loaded if needed, without reloading pipeline."""
        global current_reprompt_loaded
        
        if self.pipeline is None:
            return
        
        # Set the flag indicating whether reprompt should be available
        self.pipeline._should_load_reprompt = should_load
        current_reprompt_loaded = should_load
        
        if should_load and not hasattr(self.pipeline, '_reprompt_model'):
            print("Loading reprompt model on demand...")
            # The model will be loaded lazily when first accessed
    
    def generate_single_image(self, 
                            prompt: str,
                            negative_prompt: str,
                            width: int,
                            height: int,
                            num_inference_steps: int,
                            guidance_scale: float,
                            seed: int,
                            use_reprompt: bool,
                            use_refiner: bool,
                            refiner_steps: int,
                            auto_enhance: bool,
                            main_shift: int = 4,
                            refiner_shift: int = 1,
                            refiner_guidance: float = 1.5) -> Tuple[Optional[Image.Image], Optional[Image.Image], Dict, str]:
        """Generate a single image and return it with metadata and final prompt."""
        torch.cuda.empty_cache()
        
        # Auto enhance prompt if requested
        enhanced_prompt = prompt
        actual_use_reprompt = use_reprompt
        final_prompt_for_return = prompt  # Track what prompt to return to UI
        
        # Ensure reprompt is loaded if needed
        self._ensure_reprompt_loaded(auto_enhance)
        
        # Check if reprompt model is available
        has_reprompt = hasattr(self.pipeline, '_should_load_reprompt') and self.pipeline._should_load_reprompt
        
        # Handle prompt enhancement - ONLY if auto_enhance is True
        if has_reprompt and auto_enhance:
            # Auto enhance is enabled, enhance the prompt
            self.pipeline.to('cpu')
            if hasattr(self.pipeline, '_refiner_pipeline'):
                self.pipeline.refiner_pipeline.to('cpu')
            enhanced_prompt = self.pipeline.reprompt_model.predict(prompt)
            final_prompt_for_return = enhanced_prompt
        elif auto_enhance and not has_reprompt:
            print("Warning: Auto enhance requested but reprompt model not loaded. Using original prompt.")
            actual_use_reprompt = False
        
        # Move pipeline to GPU
        if hasattr(self.pipeline, '_refiner_pipeline'):
            self.pipeline.refiner_pipeline.to('cpu')
        self.pipeline.to('cuda')
        
        # Generate seed if random
        if seed == -1:
            seed = random.randint(100000, 999999)
        
        # Create metadata
        global current_model_type, current_offloading_settings
        metadata = {
            'model_type': current_model_type,
            'enable_dit_offloading': current_offloading_settings['dit'],
            'enable_reprompt_offloading': current_offloading_settings['reprompt'],
            'enable_refiner_offloading': current_offloading_settings['refiner'],
            'prompt': enhanced_prompt if auto_enhance else prompt,
            'original_prompt': prompt if auto_enhance else None,
            'negative_prompt': negative_prompt,
            'width': width,
            'height': height,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'use_reprompt': use_reprompt,
            'use_refiner': use_refiner,
            'refiner_steps': refiner_steps if use_refiner else None,
            'refiner_guidance': refiner_guidance if use_refiner else None,
            'auto_enhance': auto_enhance,
            'main_shift': main_shift,
            'refiner_shift': refiner_shift
        }
        
        # Generate image
        pre_refiner_image = None
        
        if use_refiner:
            # Refiner is loaded on-demand via the property accessor
            # Generate base image first without refiner
            base_image = self.pipeline(
                prompt=enhanced_prompt if auto_enhance else prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=main_shift,
                seed=seed,
                use_reprompt=False,  # Never use pipeline's internal enhancement
                use_refiner=False
            )
            pre_refiner_image = base_image
            
            # Apply refiner (loaded on-demand via property)
            self.pipeline.to('cpu')
            # Accessing refiner_pipeline property will load it if not already loaded
            self.pipeline.refiner_pipeline.to('cuda')
            
            final_image = self.pipeline.refiner_pipeline(
                image=base_image,
                prompt=enhanced_prompt if auto_enhance else prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=refiner_steps,
                guidance_scale=refiner_guidance,  # Use separate guidance for refiner
                shift=refiner_shift,
                seed=seed
            )
        else:
            final_image = self.pipeline(
                prompt=enhanced_prompt if auto_enhance else prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=main_shift,
                seed=seed,
                use_reprompt=False,  # Never use pipeline's internal enhancement
                use_refiner=False
            )
        
        self.print_peak_memory()
        
        # Return the final prompt that was actually used
        # This includes prompts enhanced by either auto_enhance or use_reprompt
        
        return final_image, pre_refiner_image, metadata, final_prompt_for_return

    def generate_images(self,
                       model_type: str,
                       enable_dit_offloading: bool,
                       enable_reprompt_offloading: bool,
                       enable_refiner_offloading: bool,
                       prompt: str,
                       negative_prompt: str,
                       width: int,
                       height: int,
                       num_inference_steps: int,
                       guidance_scale: float,
                       seed: int,
                       use_reprompt: bool,
                       use_refiner: bool,
                       refiner_steps: int,
                       auto_enhance: bool,
                       num_generations: int,
                       multi_line_prompt: bool,
                       main_shift: int = 4,
                       refiner_shift: int = 1,
                       refiner_guidance: float = 1.5) -> Tuple[List[str], str, str]:
        """Generate multiple images with proper seed handling."""
        try:
            # Ensure pipeline is loaded with user settings on first generation
            self.ensure_pipeline_loaded(model_type, enable_dit_offloading,
                                      enable_reprompt_offloading, enable_refiner_offloading,
                                      use_reprompt, auto_enhance)
            
            # Switch model/settings if needed (after initial load)
            switch_status = self.switch_model(model_type, enable_dit_offloading, 
                                            enable_reprompt_offloading, enable_refiner_offloading,
                                            use_reprompt, auto_enhance)
            print(switch_status)
            
            if self.pipeline is None:
                return [], "Pipeline not loaded. Please try again.", prompt
            
            gallery_paths = []  # Paths for gallery display
            saved_paths = []  # For status message
            final_used_prompt = prompt
            
            # Handle multi-line prompt mode
            prompts_to_process = []
            if multi_line_prompt:
                # Split by lines and filter out empty lines
                lines = [line.strip() for line in prompt.split('\n') if line.strip()]
                if lines:
                    prompts_to_process = lines
                else:
                    prompts_to_process = [prompt]  # Fallback to original if no valid lines
            else:
                prompts_to_process = [prompt]
            
            # Calculate total images to generate
            total_images = len(prompts_to_process) * num_generations
            image_counter = 0
            
            # Process each prompt
            for prompt_line in prompts_to_process:
                for gen_idx in range(num_generations):
                    image_counter += 1
                    # Calculate seed for this generation
                    if seed == -1:
                        current_seed = -1  # Will be randomized in generate_single_image
                    else:
                        # For multi-line, increment seed across all images
                        current_seed = seed + (image_counter - 1)
                
                    # Generate single image with current prompt line
                    image, pre_refiner_image, metadata, used_prompt = self.generate_single_image(
                        prompt=prompt_line,
                        negative_prompt=negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=current_seed,
                        use_reprompt=use_reprompt,
                        use_refiner=use_refiner,
                        refiner_steps=refiner_steps,
                        auto_enhance=auto_enhance,
                        main_shift=main_shift,
                        refiner_shift=refiner_shift,
                        refiner_guidance=refiner_guidance
                    )
                    
                    # Update metadata with multi-line info if applicable
                    if multi_line_prompt:
                        metadata['prompt_line_index'] = prompts_to_process.index(prompt_line)
                        metadata['total_prompt_lines'] = len(prompts_to_process)
                    
                    # Save pre-refiner image if it exists
                    if pre_refiner_image:
                        pre_path = self.image_saver.save_image(pre_refiner_image, metadata, is_pre_refiner=True)
                        print(f"Saved pre-refiner image: {pre_path}")
                    
                    # Save final image
                    path = self.image_saver.save_image(image, metadata)
                    saved_paths.append(path)
                    gallery_paths.append(path)  # Return file path for proper extension handling
                    
                    # Update final_used_prompt for single-line mode
                    if not multi_line_prompt:
                        final_used_prompt = used_prompt
                    
                    print(f"Saved image {image_counter}/{total_images}: {path}")
            
            if multi_line_prompt:
                status = f"Successfully generated {total_images} image(s) from {len(prompts_to_process)} prompt lines!\nSaved to outputs folder"
                # Don't update prompt box for multi-line mode
                return gallery_paths, status, prompt
            else:
                status = f"Successfully generated {num_generations} image(s)!\nSaved to: {', '.join(saved_paths)}"
                return gallery_paths, status, final_used_prompt
            
        except Exception as e:
            error_msg = f"Error generating images: {str(e)}"
            print(f"✗ {error_msg}")
            return [], error_msg, prompt

    def enhance_prompt(self, prompt: str, model_type: str = "regular",
                      enable_dit_offloading: bool = True,
                      enable_reprompt_offloading: bool = True,
                      enable_refiner_offloading: bool = True,
                      use_reprompt: bool = False,
                      auto_enhance: bool = False) -> Tuple[str, str]:
        """Enhance a prompt using the reprompt model."""
        try:
            torch.cuda.empty_cache()
            
            # Ensure pipeline is loaded with user settings on first use
            self.ensure_pipeline_loaded(model_type, enable_dit_offloading,
                                      enable_reprompt_offloading, enable_refiner_offloading,
                                      use_reprompt, auto_enhance)

            if self.pipeline is None:
                return prompt, "Pipeline not loaded. Please try again."
            
            # Check if reprompt model is available
            has_reprompt = hasattr(self.pipeline, '_should_load_reprompt') and self.pipeline._should_load_reprompt
            
            if not has_reprompt:
                return prompt, "Reprompt model not loaded. Enable 'Load Reprompt Model' first."
            
            self.pipeline.to('cpu')
            if hasattr(self.pipeline, '_refiner_pipeline'):
                self.pipeline.refiner_pipeline.to('cpu')

            enhanced_prompt = self.pipeline.reprompt_model.predict(prompt)
            self.print_peak_memory()
            return enhanced_prompt, "Prompt enhanced successfully!"
            
        except Exception as e:
            error_msg = f"Error enhancing prompt: {str(e)}"
            print(f"✗ {error_msg}")
            return prompt, error_msg
    
    def refine_existing_image(self,
                             model_type: str,
                             enable_dit_offloading: bool,
                             enable_reprompt_offloading: bool, 
                             enable_refiner_offloading: bool,
                             input_image: Optional[Image.Image],
                             prompt: str,
                             negative_prompt: str,
                             width: int,
                             height: int,
                             num_inference_steps: int,
                             guidance_scale: float,
                             seed: int,
                             refiner_shift: int = 1) -> Tuple[Optional[str], str, Dict]:
        """Refine an existing image using the refiner pipeline."""
        try:
            if input_image is None:
                return None, "Please upload an image to refine.", {}
            
            torch.cuda.empty_cache()
            
            # Ensure pipeline is loaded with user settings
            self.ensure_pipeline_loaded(model_type, enable_dit_offloading,
                                      enable_reprompt_offloading, enable_refiner_offloading,
                                      False, False)  # Don't need reprompt for refinement
            
            if self.pipeline is None:
                return None, "Pipeline not loaded. Please try again.", {}
            
            # Handle image resizing with smart cropping to maintain aspect ratio
            original_size = input_image.size
            if original_size != (width, height):
                # Calculate aspect ratios
                input_aspect = original_size[0] / original_size[1]
                target_aspect = width / height
                
                # Smart crop and resize to maintain aspect ratio
                if abs(input_aspect - target_aspect) > 0.01:  # If aspects differ
                    print(f"📐 Input aspect ratio ({input_aspect:.2f}) differs from target ({target_aspect:.2f})")
                    print(f"   Using smart center crop to maintain aspect ratio...")
                    
                    # Calculate dimensions for center crop
                    if input_aspect > target_aspect:
                        # Input is wider - crop width
                        new_width = int(original_size[1] * target_aspect)
                        new_height = original_size[1]
                        left = (original_size[0] - new_width) // 2
                        top = 0
                        right = left + new_width
                        bottom = new_height
                    else:
                        # Input is taller - crop height
                        new_width = original_size[0]
                        new_height = int(original_size[0] / target_aspect)
                        left = 0
                        top = (original_size[1] - new_height) // 2
                        right = new_width
                        bottom = top + new_height
                    
                    # Crop to target aspect ratio
                    input_image = input_image.crop((left, top, right, bottom))
                    print(f"   Cropped from {original_size} to {input_image.size} (center crop)")
                
                # Now resize to exact target dimensions
                if input_image.size != (width, height):
                    input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
                    print(f"   Final resize to ({width}, {height})")
            
            # Move main pipeline to CPU, refiner to GPU
            self.pipeline.to('cpu')
            # Accessing refiner_pipeline property will load it if not already loaded
            self.pipeline.refiner_pipeline.to('cuda')
            
            # Generate seed if random
            if seed == -1:
                seed = random.randint(100000, 999999)
            
            # Create metadata for the refined image
            global current_model_type, current_offloading_settings
            metadata = {
                'operation': 'image_refinement',
                'model_type': current_model_type,
                'enable_dit_offloading': current_offloading_settings['dit'],
                'enable_reprompt_offloading': current_offloading_settings['reprompt'],
                'enable_refiner_offloading': current_offloading_settings['refiner'],
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'width': width,
                'height': height,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'refiner_shift': refiner_shift,
                'original_image_size': original_size,
                'preprocessing': 'center_crop_and_resize' if original_size != (width, height) else 'none',
                'aspect_ratio_preserved': True
            }
            
            # Apply refiner
            refined_image = self.pipeline.refiner_pipeline(
                image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=refiner_shift,
                seed=seed
            )
            
            self.print_peak_memory()
            
            # Save the refined image
            path = self.image_saver.save_image(refined_image, metadata)
            print(f"Saved refined image: {path}")
            
            # Return as list for Gallery component
            return [path], f"Image refined successfully!\nSaved to: {path}", metadata
            
        except Exception as e:
            error_msg = f"Error refining image: {str(e)}"
            print(f"✗ {error_msg}")
            return [], error_msg, {}

    def open_outputs_folder(self):
        """Open the outputs folder in the system file explorer."""
        try:
            folder_path = str(OUTPUTS_DIR.absolute())
            
            if platform.system() == 'Windows':
                os.startfile(folder_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', folder_path])
            else:  # Linux
                subprocess.run(['xdg-open', folder_path])
            
            return f"Opened folder: {folder_path}"
        except Exception as e:
            return f"Error opening folder: {str(e)}"

    def save_config(self, config_name: str, **params) -> str:
        """Save current configuration."""
        if not config_name:
            return "Please enter a configuration name"
        
        # Remove None values and input_image
        clean_params = {k: v for k, v in params.items() 
                       if v is not None and k != 'input_image'}
        
        if self.config_manager.save_config(config_name, clean_params):
            return f"Configuration '{config_name}' saved successfully!"
        return "Failed to save configuration"

    def load_config(self, config_name: str) -> Tuple[Dict, str]:
        """Load a configuration."""
        if not config_name:
            return {}, "Please select a configuration"
        
        params = self.config_manager.load_config(config_name)
        if params:
            return params, f"Configuration '{config_name}' loaded successfully!"
        return {}, f"Failed to load configuration '{config_name}'"

    def get_config_list(self) -> List[str]:
        """Get list of available configurations."""
        return self.config_manager.list_configs()


def create_interface(auto_load: bool = True, use_distilled: bool = False, device: str = "cuda"):
    """Create the Gradio interface."""
    app = HunyuanImageApp(auto_load=auto_load, use_distilled=use_distilled, device=device)
    
    # Load last configuration if available
    last_config = app.config_manager.load_last_config()
    
    # Custom CSS
    css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    .model-info {
        background: var(--background-fill-secondary);
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="HunyuanImage Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## SECourses HunyuanImage 2.1 Pro V2 : https://www.patreon.com/posts/138531984")
        with gr.Tabs():           
            with gr.Tab("🖼️ Text-to-Image Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            generate_btn = gr.Button("🎨 Generate Image(s)", variant="primary", size="lg")
                            open_folder_btn = gr.Button("📁 Open Outputs Folder", variant="secondary")
                        
                        gr.Markdown("### Generation Settings")
                        
                        # Model selection
                        model_type = gr.Radio(
                            label="Model Type",
                            choices=["regular", "distilled"],
                            value=last_config.get('model_type', 'regular') if last_config else 'regular',
                            info="Regular: Higher quality (50 steps) | Distilled: Faster generation (8 steps)\nIf you want to change model restart app recommended"
                        )
                        
                        # GPU Memory Optimization Settings
                        gr.Markdown("### GPU Memory Optimization")
                        with gr.Accordion("ℹ️ How Model Loading Works", open=False):
                            gr.Markdown(
                                """
                                **Deferred Loading:**
                                - Models are loaded on first generation with your selected VRAM settings
                                - This ensures optimal memory usage based on your GPU configuration
                                - Configure these settings BEFORE your first generation
                                
                                **Available Models:**
                                - **Main DiT Model**: Core diffusion model for image generation (always loaded)
                                - **Reprompt Model**: Optional LLM for prompt enhancement (only loaded if "Use Reprompt Model" is checked)
                                - **Refiner Model**: Optional model for image refinement (loaded when refiner is used)
                                
                                **Generation Process:**
                                - **Without Reprompt/Refiner**: Single pass using main model
                                - **With Reprompt**: Enhances prompt before generation (adds ~7GB VRAM)
                                - **With Refiner**: Two-stage process:
                                  1. Base generation with main model (full steps)
                                  2. Refinement pass (default 4 steps, adjustable)
                                
                                **Memory Management:**
                                - Only one model on GPU at a time
                                - Models swap between CPU/GPU automatically
                                - Offloading moves unused models to CPU to save VRAM
                                - Reprompt model is NOT loaded unless "Use Reprompt Model" is enabled
                                
                                **Recommended Settings:**
                                - **≤12GB VRAM**: Enable all offloading, disable reprompt model
                                - **16-20GB VRAM**: Enable offloading, use reprompt carefully
                                - **≥24GB VRAM**: Disable offloading for speed, reprompt model OK
                                """
                            )
                        
                        with gr.Row():
                            enable_dit_offloading = gr.Checkbox(
                                label="DiT Offloading",
                                value=last_config.get('enable_dit_offloading', True) if last_config else True,
                                info="Move DiT model to CPU when not in use (saves VRAM)"
                            )
                            enable_reprompt_offloading = gr.Checkbox(
                                label="Reprompt Offloading",
                                value=last_config.get('enable_reprompt_offloading', True) if last_config else True,
                                info="Move reprompt model to CPU when not in use (only applies if reprompt model is enabled)"
                            )
                            enable_refiner_offloading = gr.Checkbox(
                                label="Refiner Offloading",
                                value=last_config.get('enable_refiner_offloading', True) if last_config else True,
                                info="Move refiner model to CPU when not in use"
                            )
                        
                        prompt = gr.Textbox(
                            label="Prompt",
                            lines=3,
                            value=last_config.get('prompt', "A cute cartoon penguin") if last_config else "A cute cartoon penguin",
                            placeholder="Enter your prompt here. With Multi-Line mode, each line becomes a separate generation."
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            lines=2,
                            value=last_config.get('negative_prompt', "") if last_config else ""
                        )
                        
                        # Aspect ratio presets
                        gr.Markdown("### Aspect Ratio Presets")
                        with gr.Row():
                            aspect_ratio = gr.Radio(
                                choices=["Custom", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", "2:3", "3:2"],
                                value="Custom",
                                label="Aspect Ratio",
                                interactive=True
                            )
                        
                        with gr.Row():
                            width = gr.Slider(
                                minimum=512, maximum=3072, step=32,
                                value=last_config.get('width', 2048) if last_config else 2048,
                                label="Width"
                            )
                            height = gr.Slider(
                                minimum=512, maximum=3072, step=32,
                                value=last_config.get('height', 2048) if last_config else 2048,
                                label="Height"
                            )
                        
                        # Aspect ratio presets (all divisible by 32)
                        aspect_ratios = {
                            "1:1": (2048, 2048),    # Square
                            "16:9": (2560, 1440),   # Widescreen (adjusted from 1536 to 1440 for divisibility)
                            "9:16": (1440, 2560),   # Portrait widescreen
                            "4:3": (2304, 1728),    # Classic TV (adjusted from 1792 to 1728)
                            "3:4": (1728, 2304),    # Portrait classic
                            "21:9": (2688, 1152),   # Ultrawide
                            "9:21": (1152, 2688),   # Portrait ultrawide
                            "2:3": (1664, 2496),    # Portrait photo
                            "3:2": (2496, 1664),    # Landscape photo
                        }
                        
                        def update_dimensions(aspect):
                            if aspect in aspect_ratios:
                                w, h = aspect_ratios[aspect]
                                return gr.update(value=w), gr.update(value=h)
                            return gr.update(), gr.update()
                        
                        aspect_ratio.change(
                            fn=update_dimensions,
                            inputs=[aspect_ratio],
                            outputs=[width, height]
                        )
                        
                        # Update aspect ratio to Custom when manual sliders are changed
                        width.change(fn=lambda: "Custom", outputs=[aspect_ratio])
                        height.change(fn=lambda: "Custom", outputs=[aspect_ratio])
                        
                        with gr.Row():
                            num_inference_steps = gr.Slider(
                                minimum=1, maximum=100, step=1,
                                value=last_config.get('num_inference_steps', 50) if last_config else 50,
                                label="Inference Steps"
                            )
                            guidance_scale = gr.Slider(
                                minimum=1.0, maximum=10.0, step=0.1,
                                value=last_config.get('guidance_scale', 3.5) if last_config else 3.5,
                                label="Guidance Scale"
                            )
                        
                        # Shift parameters for timestep scheduling
                        with gr.Row():
                            main_shift = gr.Slider(
                                minimum=1, maximum=10, step=1,
                                value=last_config.get('main_shift', 4) if last_config else 4,
                                label="Main Model Shift",
                                info="Timestep shift for main model (default: 4, higher = more denoising)"
                            )
                            refiner_shift = gr.Slider(
                                minimum=1, maximum=10, step=1,
                                value=last_config.get('refiner_shift', 1) if last_config else 1,
                                label="Refiner Shift",
                                info="Timestep shift for refiner (default: 1, lower = less noise)"
                            )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Images")
                        generated_images = gr.Gallery(
                            label="Generated Images",
                            show_label=False,
                            elem_id="gallery",
                            columns=2,
                            rows=2,
                            height=400
                        )
                        generation_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Models will be loaded on first generation with your selected VRAM settings"
                        )
                        
                        gr.Markdown("### Generation Controls")
                        with gr.Row():
                            seed = gr.Number(
                                label="Seed",
                                value=last_config.get('seed', -1) if last_config else -1,
                                precision=0
                            )
                            num_generations = gr.Slider(
                                minimum=1, maximum=20, step=1,
                                value=last_config.get('num_generations', 1) if last_config else 1,
                                label="Number of Images"
                            )
                        
                        gr.Markdown("### Model Features")
                        with gr.Accordion("ℹ️ Understanding Reprompt Options", open=False):
                            gr.Markdown(
                                """
                                **Load Reprompt Model vs Auto Enhance Prompt:**
                                
                                **Load Reprompt Model** (checkbox below):
                                - Controls whether the reprompt LLM model is loaded into memory
                                - Just loading it does NOT automatically enhance prompts
                                - Must be enabled to use ANY prompt enhancement features
                                - Adds ~7GB VRAM requirement when enabled
                                - If disabled, no prompt enhancement is possible
                                
                                **Auto Enhance Prompt** (checkbox below):
                                - Only works if "Load Reprompt Model" is enabled
                                - This is what actually triggers automatic enhancement
                                - When checked: Automatically enhances your prompt before every generation
                                - When unchecked: Uses your original prompt as-is
                                - Enhanced prompt will be shown in the prompt box after generation
                                
                                **Manual Enhancement** (in Prompt Enhancement tab):
                                - Requires "Load Reprompt Model" to be enabled
                                - Works independently of "Auto Enhance Prompt"
                                - Lets you enhance prompts manually and see results before using them
                                - Good for testing and fine-tuning prompts
                                
                                **Example Workflows:**
                                1. **No enhancement**: Both checkboxes unchecked
                                2. **Manual only**: Enable "Load Reprompt Model", use Enhancement tab
                                3. **Auto enhancement**: Enable both "Load Reprompt Model" AND "Auto Enhance Prompt"
                                """
                            )
                        
                        with gr.Row():
                            use_reprompt = gr.Checkbox(
                                label="Load Reprompt Model",
                                value=last_config.get('use_reprompt', True) if last_config else True,
                                info="Load the reprompt LLM model into memory (required for enhancement features, +7GB VRAM, doesn't auto-enhance)"
                            )
                            auto_enhance = gr.Checkbox(
                                label="Auto Enhance Prompt",
                                value=last_config.get('auto_enhance', False) if last_config else False,
                                info="Automatically enhance every prompt before generation (requires 'Load Reprompt Model' to be enabled)"
                            )
                        
                        with gr.Row():
                            multi_line_prompt = gr.Checkbox(
                                label="Multi-Line Prompt",
                                value=last_config.get('multi_line_prompt', False) if last_config else False,
                                info="Process each line of the prompt as a separate generation (won't return enhanced prompts to box)"
                            )
                        
                        with gr.Accordion("ℹ️ Multi-Line Prompt Mode", open=False):
                            gr.Markdown(
                                """
                                **How Multi-Line Prompt Works:**
                                
                                When enabled, each line in your prompt box becomes a separate image generation:
                                - Line 1 → Image(s) with prompt from line 1
                                - Line 2 → Image(s) with prompt from line 2
                                - And so on...
                                
                                **Example:**
                                ```
                                A majestic mountain at sunrise
                                A serene lake with reflection
                                A dense forest in autumn
                                ```
                                With "Number of Images" = 2, this generates 6 images total (2 per line).
                                
                                **Features:**
                                - Each line gets its own prompt enhancement (if enabled)
                                - Seeds increment across all images for variety
                                - Metadata tracks which line each image came from
                                - Empty lines are automatically skipped
                                - Enhanced prompts are NOT returned to the prompt box
                                
                                **Use Cases:**
                                - Batch generate different concepts
                                - Test variations of similar prompts
                                - Create themed image sets efficiently
                                """
                            )
                        
                        with gr.Row():
                            use_refiner = gr.Checkbox(
                                label="Use Refiner",
                                value=last_config.get('use_refiner', False) if last_config else False
                            )
                            refiner_steps = gr.Slider(
                                minimum=1, maximum=20, step=1,
                                value=last_config.get('refiner_steps', 4) if last_config else 4,
                                label="Refiner Steps",
                                info="Steps (when refiner enabled)"
                            )
                        
                        with gr.Row():
                            refiner_guidance = gr.Slider(
                                minimum=0.5, maximum=5.0, step=0.1,
                                value=last_config.get('refiner_guidance', 1.5) if last_config else 1.5,
                                label="Refiner Guidance Scale",
                                info="Try different values to find what works best for your images"
                            )
                        
                        # Configuration management
                        gr.Markdown("### Configuration Management")
                        with gr.Row():
                            config_dropdown = gr.Dropdown(
                                label="Load Configuration",
                                choices=app.get_config_list(),
                                interactive=True
                            )
                            refresh_btn = gr.Button("🔄", size="sm")
                        
                        with gr.Row():
                            config_name_input = gr.Textbox(
                                label="Config Name",
                                placeholder="Enter config name to save"
                            )
                            save_config_btn = gr.Button("💾 Save Config", size="sm")
            
            # Prompt Enhancement Tab
            with gr.Tab("✨ Prompt Enhancement"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Manual Prompt Enhancement")
                        gr.Markdown(
                            """
                            **Note:** This tab requires "Load Reprompt Model" to be enabled in the Text-to-Image tab.
                            
                            Use this to manually enhance prompts and review them before generation.
                            The enhanced prompt can be copied and used in the main generation tab.
                            This works independently of "Auto Enhance Prompt".
                            """
                        )
                        
                        original_prompt = gr.Textbox(
                            label="Original Prompt",
                            lines=4,
                            value="A cat sitting on a table"
                        )
                        
                        enhance_btn = gr.Button("✨ Enhance Prompt", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Enhanced Prompt")
                        enhanced_prompt = gr.Textbox(
                            label="Enhanced Prompt",
                            lines=6,
                            interactive=False
                        )
                        enhancement_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready to enhance (requires 'Load Reprompt Model' to be enabled)"
                        )
            
            # Image Refinement Tab
            with gr.Tab("🔧 Image Refinement"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Refinement Settings")
                        gr.Markdown(
                            """
                            **Upload any image to enhance it with the refiner model.**
                            
                            The refiner can improve details, quality, and apply your prompt guidance to existing images.
                            Works best with images generated by HunyuanImage, but can refine any image.
                            """
                        )
                        
                        with gr.Row():
                            refine_btn = gr.Button("🔧 Refine Image", variant="primary", size="lg")
                            refine_open_folder_btn = gr.Button("📁 Open Outputs Folder", variant="secondary")
                        
                        auto_aspect_ratio = gr.Checkbox(
                            label="Auto-detect aspect ratio",
                            value=last_config.get('auto_aspect_ratio', True) if last_config else True,
                            info="Automatically set output dimensions to match input image aspect ratio"
                        )
                        
                        input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=300
                        )
                        
                        refine_prompt = gr.Textbox(
                            label="Refinement Prompt",
                            placeholder="Describe what you want in the refined image",
                            lines=3,
                            value=last_config.get('refine_prompt', "High quality, detailed, sharp focus") if last_config else "High quality, detailed, sharp focus"
                        )
                        
                        refine_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What to avoid in the refinement",
                            lines=2,
                            value=last_config.get('refine_negative_prompt', "") if last_config else ""
                        )
                        
                        # Aspect ratio presets for refinement
                        gr.Markdown("### Output Dimensions")
                        with gr.Row():
                            refine_aspect_ratio = gr.Radio(
                                choices=["Custom", "1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "9:21", "2:3", "3:2"],
                                value="Custom",
                                label="Aspect Ratio",
                                interactive=True
                            )
                        
                        with gr.Row():
                            refine_width = gr.Slider(
                                minimum=512, maximum=3072, step=32,
                                value=last_config.get('refine_width', 2048) if last_config else 2048,
                                label="Width"
                            )
                            refine_height = gr.Slider(
                                minimum=512, maximum=3072, step=32,
                                value=last_config.get('refine_height', 2048) if last_config else 2048,
                                label="Height"
                            )
                        
                        def update_refine_dimensions(aspect):
                            if aspect in aspect_ratios:
                                w, h = aspect_ratios[aspect]
                                return gr.update(value=w), gr.update(value=h)
                            return gr.update(), gr.update()
                        
                        refine_aspect_ratio.change(
                            fn=update_refine_dimensions,
                            inputs=[refine_aspect_ratio],
                            outputs=[refine_width, refine_height]
                        )
                        
                        # Update aspect ratio to Custom when manual sliders are changed
                        refine_width.change(fn=lambda: "Custom", outputs=[refine_aspect_ratio])
                        refine_height.change(fn=lambda: "Custom", outputs=[refine_aspect_ratio])
                        
                        with gr.Row():
                            refine_steps = gr.Slider(
                                minimum=1, maximum=20, step=1,
                                value=last_config.get('refine_steps', 4) if last_config else 4,
                                label="Refinement Steps",
                                info="More steps = more refinement (default: 4)"
                            )
                            refine_guidance = gr.Slider(
                                minimum=0.5, maximum=5.0, step=0.1,
                                value=last_config.get('refine_guidance', 1.5) if last_config else 1.5,
                                label="Guidance Scale",
                                info="How strongly to follow the prompt"
                            )
                        
                        with gr.Row():
                            refine_seed = gr.Number(
                                label="Seed",
                                value=last_config.get('refine_seed', -1) if last_config else -1,
                                precision=0,
                                info="Random seed for reproducibility (-1 for random)"
                            )
                            refine_shift = gr.Slider(
                                minimum=1, maximum=10, step=1,
                                value=last_config.get('refine_shift', 1) if last_config else 1,
                                label="Refiner Shift",
                                info="Timestep shift for refiner (default: 1, lower = less noise)"
                            )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Refined Image")
                        refined_image_gallery = gr.Gallery(
                            label="Refined Image",
                            show_label=False,
                            elem_id="refined_gallery",
                            columns=1,
                            rows=1,
                            height=400
                        )
                        refinement_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready to refine"
                        )
                        
                        # Configuration management for refinement
                        gr.Markdown("### Refinement Config Management")
                        with gr.Row():
                            refine_config_dropdown = gr.Dropdown(
                                label="Load Configuration",
                                choices=app.get_config_list(),
                                interactive=True
                            )
                            refine_refresh_btn = gr.Button("🔄", size="sm")
                        
                        with gr.Row():
                            refine_config_name_input = gr.Textbox(
                                label="Config Name",
                                placeholder="Enter config name to save"
                            )
                            refine_save_config_btn = gr.Button("💾 Save Config", size="sm")
                        
                        gr.Markdown("### Refinement Info")
                        with gr.Accordion("📐 How Image Resizing Works", open=False):
                            gr.Markdown(
                                """
                                **Smart Aspect Ratio Handling:**
                                
                                When your input image aspect ratio doesn't match the output dimensions:
                                1. **Center Crop**: The image is automatically center-cropped to match the target aspect ratio
                                2. **Resize**: Then scaled to the exact output dimensions
                                3. **No Distortion**: Your image won't be stretched or squashed
                                
                                **Example:**
                                - Input: 1920×1080 (16:9) → Output: 1024×1024 (1:1)
                                - Process: Centers and crops to 1080×1080, then resizes to 1024×1024
                                
                                **Tips:**
                                - Use aspect ratio presets that match your input for no cropping
                                - Important content should be centered in your image
                                - Check the console for details about cropping applied
                                """
                            )
                        
                        with gr.Accordion("ℹ️ How Image Refinement Works", open=False):
                            gr.Markdown(
                                """
                                **The Refiner Model:**
                                - Specialized model for enhancing image quality and details
                                - Can be applied to any image, not just generated ones
                                - Uses your prompt to guide the refinement process
                                - Typically uses fewer steps than generation (4 steps default)
                                
                                **Best Practices:**
                                - **For generated images**: Use to add final polish and details
                                - **For photos/artwork**: Enhance quality while preserving style
                                - **Prompt guidance**: Describe desired improvements clearly
                                - **Steps**: 4 steps usually sufficient, more for stronger changes
                                - **Guidance scale**: Lower values (1-2) preserve original more
                                
                                **Memory Usage:**
                                - Refiner is loaded on-demand when first used
                                - With offloading enabled, swaps with main model automatically
                                - Adds ~4-6GB VRAM when active
                                
                                **Output:**
                                - Refined images are saved to the outputs folder
                                - Metadata includes all refinement parameters
                                - Original image is preserved (not overwritten)
                                """
                            )
        
        # Function to auto-detect and set closest aspect ratio for refinement
        def auto_detect_aspect_ratio(image, auto_detect_enabled):
            """Detect input image aspect ratio and set closest matching output dimensions."""
            if not auto_detect_enabled or image is None:
                return gr.update(), gr.update(), gr.update()
            
            # Get input image dimensions
            width, height = image.size
            input_aspect = width / height
            
            # Find closest matching aspect ratio from presets
            closest_ratio = "Custom"
            min_diff = float('inf')
            
            for ratio_name, (preset_w, preset_h) in aspect_ratios.items():
                preset_aspect = preset_w / preset_h
                diff = abs(input_aspect - preset_aspect)
                if diff < min_diff:
                    min_diff = diff
                    closest_ratio = ratio_name
                    closest_width = preset_w
                    closest_height = preset_h
            
            # If very close to a preset (within 5%), use it exactly
            if min_diff < 0.05:
                print(f"🎯 Auto-detected aspect ratio: {closest_ratio} ({closest_width}×{closest_height})")
                return gr.update(value=closest_ratio), gr.update(value=closest_width), gr.update(value=closest_height)
            else:
                # Calculate custom dimensions that match input aspect exactly
                # Use common resolutions as base
                if width > height:  # Landscape
                    if width >= 2048:
                        new_width = 2560 if input_aspect > 1.5 else 2048
                    else:
                        new_width = 1920 if input_aspect > 1.5 else 1536
                    new_height = round(new_width / input_aspect / 32) * 32  # Round to nearest 32
                else:  # Portrait or square
                    if height >= 2048:
                        new_height = 2560 if input_aspect < 0.67 else 2048
                    else:
                        new_height = 1920 if input_aspect < 0.67 else 1536
                    new_width = round(new_height * input_aspect / 32) * 32  # Round to nearest 32
                
                # Clamp to valid range
                new_width = max(512, min(3072, new_width))
                new_height = max(512, min(3072, new_height))
                
                print(f"🎯 Auto-detected custom dimensions: {new_width}×{new_height} (aspect ratio: {input_aspect:.2f})")
                return gr.update(value="Custom"), gr.update(value=new_width), gr.update(value=new_height)
        
        # Event handlers
        generate_btn.click(
            fn=app.generate_images,
            inputs=[
                model_type, enable_dit_offloading, enable_reprompt_offloading, enable_refiner_offloading,
                prompt, negative_prompt, width, height, num_inference_steps,
                guidance_scale, seed, use_reprompt, use_refiner, refiner_steps, auto_enhance,
                num_generations, multi_line_prompt, main_shift, refiner_shift, refiner_guidance
            ],
            outputs=[generated_images, generation_status, prompt]  # Return final prompt to prompt box
        )
        
        enhance_btn.click(
            fn=app.enhance_prompt,
            inputs=[original_prompt, model_type, enable_dit_offloading, 
                   enable_reprompt_offloading, enable_refiner_offloading, use_reprompt, auto_enhance],
            outputs=[enhanced_prompt, enhancement_status]
        )
        
        open_folder_btn.click(
            fn=app.open_outputs_folder,
            outputs=[generation_status]
        )
        
        # Image Refinement Event Handlers
        refine_btn.click(
            fn=app.refine_existing_image,
            inputs=[
                model_type, enable_dit_offloading, enable_reprompt_offloading, enable_refiner_offloading,
                input_image, refine_prompt, refine_negative_prompt, refine_width, refine_height,
                refine_steps, refine_guidance, refine_seed, refine_shift
            ],
            outputs=[refined_image_gallery, refinement_status, gr.State()]  # Using State for metadata
        )
        
        refine_open_folder_btn.click(
            fn=app.open_outputs_folder,
            outputs=[refinement_status]
        )
        
        # Auto-detect aspect ratio when image is uploaded
        input_image.change(
            fn=auto_detect_aspect_ratio,
            inputs=[input_image, auto_aspect_ratio],
            outputs=[refine_aspect_ratio, refine_width, refine_height]
        )
        
        # Also trigger when checkbox is toggled
        auto_aspect_ratio.change(
            fn=auto_detect_aspect_ratio,
            inputs=[input_image, auto_aspect_ratio],
            outputs=[refine_aspect_ratio, refine_width, refine_height]
        )
        
        # Config management handlers
        def save_and_refresh(config_name, model_type, enable_dit_offloading, enable_reprompt_offloading,
                            enable_refiner_offloading, prompt, negative_prompt, aspect_ratio, width, height,
                            num_inference_steps, guidance_scale, seed, use_reprompt,
                            use_refiner, refiner_steps, auto_enhance, multi_line_prompt, num_generations,
                            main_shift, refiner_shift, refiner_guidance):
            params = {
                'model_type': model_type,
                'enable_dit_offloading': enable_dit_offloading,
                'enable_reprompt_offloading': enable_reprompt_offloading,
                'enable_refiner_offloading': enable_refiner_offloading,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'aspect_ratio': aspect_ratio,
                'width': width,
                'height': height,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'use_reprompt': use_reprompt,
                'use_refiner': use_refiner,
                'refiner_steps': refiner_steps,
                'auto_enhance': auto_enhance,
                'multi_line_prompt': multi_line_prompt,
                'num_generations': num_generations,
                'main_shift': main_shift,
                'refiner_shift': refiner_shift,
                'refiner_guidance': refiner_guidance
            }
            status = app.save_config(config_name, **params)
            configs = app.get_config_list()
            return status, gr.update(choices=configs, value=config_name if config_name in configs else None)
        
        save_config_btn.click(
            fn=save_and_refresh,
            inputs=[
                config_name_input, model_type, enable_dit_offloading, enable_reprompt_offloading,
                enable_refiner_offloading, prompt, negative_prompt, aspect_ratio, width, height,
                num_inference_steps, guidance_scale, seed, use_reprompt,
                use_refiner, refiner_steps, auto_enhance, multi_line_prompt, num_generations,
                main_shift, refiner_shift, refiner_guidance
            ],
            outputs=[generation_status, config_dropdown]
        )
        
        def load_and_update(config_name):
            params, status = app.load_config(config_name)
            if params:
                # Determine aspect ratio from width/height if not stored
                stored_aspect = params.get('aspect_ratio', 'Custom')
                if stored_aspect == 'Custom':
                    # Check if dimensions match any preset
                    w, h = params.get('width', 2048), params.get('height', 2048)
                    for ratio, (preset_w, preset_h) in aspect_ratios.items():
                        if w == preset_w and h == preset_h:
                            stored_aspect = ratio
                            break
                
                return (
                    params.get('model_type', 'regular'),
                    params.get('enable_dit_offloading', True),
                    params.get('enable_reprompt_offloading', True),
                    params.get('enable_refiner_offloading', True),
                    params.get('prompt', ''),
                    params.get('negative_prompt', ''),
                    stored_aspect,
                    params.get('width', 2048),
                    params.get('height', 2048),
                    params.get('num_inference_steps', 50),
                    params.get('guidance_scale', 3.5),
                    params.get('seed', -1),
                    params.get('use_reprompt', True),
                    params.get('use_refiner', False),
                    params.get('refiner_steps', 4),
                    params.get('auto_enhance', False),
                    params.get('multi_line_prompt', False),
                    params.get('num_generations', 1),
                    params.get('main_shift', 4),
                    params.get('refiner_shift', 1),
                    params.get('refiner_guidance', 1.5),
                    status
                )
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), 
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), status
            )
        
        config_dropdown.change(
            fn=load_and_update,
            inputs=[config_dropdown],
            outputs=[
                model_type, enable_dit_offloading, enable_reprompt_offloading, enable_refiner_offloading,
                prompt, negative_prompt, aspect_ratio, width, height, num_inference_steps,
                guidance_scale, seed, use_reprompt, use_refiner, refiner_steps, auto_enhance,
                multi_line_prompt, num_generations, main_shift, refiner_shift, refiner_guidance, generation_status
            ]
        )
        
        refresh_btn.click(
            fn=lambda: gr.update(choices=app.get_config_list()),
            outputs=[config_dropdown]
        )
        
        # Refinement config management handlers
        def save_refine_config_and_refresh(config_name, auto_aspect_ratio, refine_prompt, refine_negative_prompt, 
                                          refine_width, refine_height, refine_steps, 
                                          refine_guidance, refine_seed, refine_shift):
            params = {
                'auto_aspect_ratio': auto_aspect_ratio,
                'refine_prompt': refine_prompt,
                'refine_negative_prompt': refine_negative_prompt,
                'refine_width': refine_width,
                'refine_height': refine_height,
                'refine_steps': refine_steps,
                'refine_guidance': refine_guidance,
                'refine_seed': refine_seed,
                'refine_shift': refine_shift
            }
            status = app.save_config(config_name, **params)
            configs = app.get_config_list()
            return status, gr.update(choices=configs, value=config_name if config_name in configs else None)
        
        refine_save_config_btn.click(
            fn=save_refine_config_and_refresh,
            inputs=[
                refine_config_name_input, auto_aspect_ratio, refine_prompt, refine_negative_prompt,
                refine_width, refine_height, refine_steps, refine_guidance, 
                refine_seed, refine_shift
            ],
            outputs=[refinement_status, refine_config_dropdown]
        )
        
        def load_refine_config(config_name):
            params, status = app.load_config(config_name)
            if params:
                return (
                    params.get('auto_aspect_ratio', True),
                    params.get('refine_prompt', 'High quality, detailed, sharp focus'),
                    params.get('refine_negative_prompt', ''),
                    params.get('refine_width', 2048),
                    params.get('refine_height', 2048),
                    params.get('refine_steps', 4),
                    params.get('refine_guidance', 1.5),
                    params.get('refine_seed', -1),
                    params.get('refine_shift', 1),
                    status
                )
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), status
            )
        
        refine_config_dropdown.change(
            fn=load_refine_config,
            inputs=[refine_config_dropdown],
            outputs=[
                auto_aspect_ratio, refine_prompt, refine_negative_prompt, refine_width, refine_height,
                refine_steps, refine_guidance, refine_seed, refine_shift, refinement_status
            ]
        )
        
        refine_refresh_btn.click(
            fn=lambda: gr.update(choices=app.get_config_list()),
            outputs=[refine_config_dropdown]
        )
        
        gr.Markdown(
            """
            ### 📝 Features
            - **Multi-generation**: Generate multiple images with sequential seeds
            - **Auto-save**: All images saved to outputs folder with metadata
            - **Config Management**: Save and load your favorite settings
            - **Smart Prompt Enhancement**: 
              - Optional reprompt model loading (saves 7GB VRAM when disabled)
              - Separate control: Load model vs Auto-enhance
              - Manual enhancement tab for testing and refinement
            - **Image Refinement**: Standalone tab to refine any image with custom prompts
            - **Pre-refiner Images**: Saves both refined and pre-refined versions
            - **GPU Memory Optimization**: Configurable model offloading to reduce VRAM usage
            
            ### 💡 Quick Tips
            - **Low VRAM?** Keep "Load Reprompt Model" disabled to save 7GB
            - **Want better prompts?** Enable "Load Reprompt Model" first, then enable "Auto Enhance" or use manual tab
            - **Testing prompts?** Use the Prompt Enhancement tab to preview enhanced versions
            """,
            elem_classes="model-info"
        )
    
    return demo


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch HunyuanImage Gradio App")
    parser.add_argument("--share", action="store_true", help="Enable Gradio live share")
    parser.add_argument("--no-auto-load", action="store_true", help="Disable auto-loading pipeline on startup")
    parser.add_argument("--use-distilled", action="store_true", help="Use distilled model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Create and launch the interface
    auto_load = not args.no_auto_load
    demo = create_interface(auto_load=auto_load, use_distilled=args.use_distilled, device=args.device)
    
    print("🚀 Starting HunyuanImage Gradio App...")
    print(f"🔧 Deferred model loading: Models will be loaded on first use with user-selected VRAM settings")
    print(f"🎯 Default model type: {'Distilled' if args.use_distilled else 'Regular'}")
    print(f"💻 Device: {args.device}")
    print(f"🌐 Share mode: {'Enabled' if args.share else 'Disabled'}")
    print("⚠️  Make sure you have the required model checkpoints in the 'ckpts' folder!")
    
    demo.launch(
        share=args.share,
        inbrowser=True,  # Open in browser by default
        show_error=True,
        quiet=False
    )