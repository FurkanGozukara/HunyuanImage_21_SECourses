import os
IS_SPACE = True

if IS_SPACE:
    import spaces


import sys
import warnings
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict
import torch

def space_context(duration: int):
    if IS_SPACE:
        return spaces.GPU(duration=duration)
    return lambda x: x

@space_context(duration=120)
def test_env():
    assert torch.cuda.is_available()

    try:
        import flash_attn
    except ImportError:
        print("Flash-attn not found, installing...")
        os.system("pip install flash-attn==2.7.3 --no-build-isolation")

    else:
        print("Flash-attn found, skipping installation...")
test_env()

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


BASE_DIR = os.environ.get('HUNYUANIMAGE_V2_1_MODEL_ROOT', './ckpts')

class CheckpointDownloader:
    """Handles downloading of all required checkpoints for HunyuanImage."""
    
    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        print(f'Downloading checkpoints to: {self.base_dir}')
        
        # Define all required checkpoints
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
                # Use modelscope for Chinese models
                return self._download_with_modelscope(config, progress_callback)
            else:
                # Use huggingface_hub for other models
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
            # Use subprocess to call modelscope CLI
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


@space_context(duration=2000)
def load_pipeline(use_distilled: bool = False, device: str = "cuda"):
    """Load the HunyuanImage pipeline (only load once, refiner and reprompt are accessed from it)."""
    try:
        assert not use_distilled # use_distilled is a placeholder for the future

        print(f"Loading HunyuanImage pipeline (distilled={use_distilled})...")
        model_name = "hunyuanimage-v2.1-distilled" if use_distilled else "hunyuanimage-v2.1"
        pipeline = HunyuanImagePipeline.from_pretrained(
            model_name=model_name,
            device=device,
            enable_dit_offloading=True,
            enable_reprompt_model_offloading=True,
            enable_refiner_offloading=True
        )
        pipeline.to('cpu')
        refiner_pipeline = pipeline.refiner_pipeline
        refiner_pipeline.text_encoder = pipeline.text_encoder
        refiner_pipeline.to('cpu')
        reprompt_model = pipeline.reprompt_model

        print("✓ Pipeline loaded successfully")
        return pipeline
    except Exception as e:
        error_msg = f"Error loading pipeline: {str(e)}"
        print(f"✗ {error_msg}")
        raise


# if IS_SPACE:
#     downloader = CheckpointDownloader()
#     downloader.download_all_checkpoints()

pipeline = load_pipeline(use_distilled=False, device="cuda")
class HunyuanImageApp:

    @space_context(duration=290)
    def __init__(self, auto_load: bool = True, use_distilled: bool = False, device: str = "cuda"):
        """Initialize the HunyuanImage Gradio app."""
        global pipeline

        self.pipeline = pipeline
        self.current_use_distilled = None


    def print_peak_memory(self):
        import torch
        stats = torch.cuda.memory_stats()
        peak_bytes_requirement = stats["allocated_bytes.all.peak"]
        print(f"Before refiner Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")

    @space_context(duration=300)
    def generate_image(self, 
                      prompt: str,
                      negative_prompt: str,
                      width: int,
                      height: int,
                      num_inference_steps: int,
                      guidance_scale: float,
                      seed: int,
                      use_reprompt: bool,
                      use_refiner: bool,
                      # use_distilled: bool
                      ) -> Tuple[Optional[Image.Image], str]:
        """Generate an image using the HunyuanImage pipeline."""
        try:
            torch.cuda.empty_cache()

            if self.pipeline is None:
                return None, "Pipeline not loaded. Please try again."


            if hasattr(self.pipeline, '_refiner_pipeline'):
                self.pipeline.refiner_pipeline.to('cpu')
            self.pipeline.to('cuda')
            if seed == -1:
                import random
                seed = random.randint(100000, 999999)

            # Generate image
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                use_reprompt=use_reprompt,
                use_refiner=use_refiner
            )
            self.print_peak_memory()
            return image, "Image generated successfully!"
                
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            print(f"✗ {error_msg}")
            return None, error_msg

    @space_context(duration=300)
    def enhance_prompt(self, prompt: str, # use_distilled: bool
                       ) -> Tuple[str, str]:
        """Enhance a prompt using the reprompt model."""
        try:
            torch.cuda.empty_cache()

            # Load pipeline if needed
            if self.pipeline is None:
                return prompt, "Pipeline not loaded. Please try again."
            
            self.pipeline.to('cpu')
            if hasattr(self.pipeline, '_refiner_pipeline'):
                self.pipeline.refiner_pipeline.to('cpu')

            # Use reprompt model from the main pipeline
            enhanced_prompt = self.pipeline.reprompt_model.predict(prompt)
            self.print_peak_memory()
            return enhanced_prompt, "Prompt enhanced successfully!"
            
        except Exception as e:
            error_msg = f"Error enhancing prompt: {str(e)}"
            print(f"✗ {error_msg}")
            return prompt, error_msg

    @space_context(duration=300)
    def refine_image(self, 
                    image: Image.Image,
                    prompt: str,
                    negative_prompt: str,
                    width: int,
                    height: int,
                    num_inference_steps: int,
                    guidance_scale: float,
                    seed: int) -> Tuple[Optional[Image.Image], str]:
        """Refine an image using the refiner pipeline."""
        try:
            if image is None:
                return None, "Please upload an image to refine."

            torch.cuda.empty_cache()

            # Resize image to target dimensions if needed
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)

            self.pipeline.to('cpu')
            self.pipeline.refiner_pipeline.to('cuda')
            if seed == -1:
                import random
                seed = random.randint(100000, 999999)
            
            # Use refiner from the main pipeline
            refined_image = self.pipeline.refiner_pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                shift=5,
                seed=seed
            )
            self.print_peak_memory()
            return refined_image, "Image refined successfully!"
            
        except Exception as e:
            error_msg = f"Error refining image: {str(e)}"
            print(f"✗ {error_msg}")
            return None, error_msg
    
    
    def download_single_checkpoint(self, checkpoint_name: str) -> Tuple[bool, str]:
        """Download a single checkpoint."""
        try:
            success, message = self.downloader.download_checkpoint(checkpoint_name)
            return success, message
        except Exception as e:
            return False, f"Download error: {str(e)}"
    
    def download_all_checkpoints(self) -> Tuple[bool, str, Dict[str, any]]:
        """Download all missing checkpoints."""
        try:
            success, message, results = self.downloader.download_all_checkpoints()
            return success, message, results
        except Exception as e:
            return False, f"Download error: {str(e)}", {}

def create_interface(auto_load: bool = True, use_distilled: bool = False, device: str = "cuda"):
    """Create the Gradio interface."""
    app = HunyuanImageApp(auto_load=auto_load, use_distilled=use_distilled, device=device)
    
    # Custom CSS for better styling with dark mode support
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .model-info {
        background: var(--background-fill-secondary);
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        color: var(--body-text-color);
    }
    .model-info h1, .model-info h2, .model-info h3 {
        color: var(--body-text-color) !important;
    }
    .model-info p, .model-info li {
        color: var(--body-text-color) !important;
    }
    .model-info strong {
        color: var(--body-text-color) !important;
    }
    """
    
    with gr.Blocks(css=css, title="HunyuanImage Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎨 HunyuanImage 2.1 Pipeline
            **HunyuanImage-2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation​**
            
            This app provides three main functionalities:
            1. **Text-to-Image Generation**: Generate high-quality images from text prompts
            2. **Prompt Enhancement**: Improve your prompts using MLLM reprompting
            3. **Image Refinement**: Enhance existing images with the refiner model
            """,
            elem_classes="model-info"
        )
        
        with gr.Tabs():
            # Tab 1: Text-to-Image Generation
            with gr.Tab("🖼️ Text-to-Image Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Generation Settings")
                        gr.Markdown("**Model**: HunyuanImage v2.1 (Non-distilled)")
                        
                        # use_distilled = gr.Checkbox(
                        #     label="Use Distilled Model",
                        #     value=False,
                        #     info="Faster generation with slightly lower quality"
                        # )
                        use_distilled = False
                        
                        prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="",
                            lines=3,
                            value="A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word “Tencent” on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."
                        )
                        
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="",
                            lines=2,
                            value=""
                        )
                        
                        with gr.Row():
                            width = gr.Slider(
                                minimum=512, maximum=2048, step=64, value=2048,
                                label="Width", info="Image width in pixels"
                            )
                            height = gr.Slider(
                                minimum=512, maximum=2048, step=64, value=2048,
                                label="Height", info="Image height in pixels"
                            )
                        
                        with gr.Row():
                            num_inference_steps = gr.Slider(
                                minimum=10, maximum=100, step=5, value=50,
                                label="Inference Steps", info="More steps = better quality, slower generation"
                            )
                            guidance_scale = gr.Slider(
                                minimum=1.0, maximum=10.0, step=0.1, value=3.5,
                                label="Guidance Scale", info="How closely to follow the prompt"
                            )
                        
                        with gr.Row():
                            seed = gr.Number(
                                label="Seed", value=-1, precision=0,
                                info="Random seed for reproducibility. (-1 for random seed)"
                            )
                            use_reprompt = gr.Checkbox(
                                label="Use Reprompt", value=True,
                                info="Enhance prompt automatically"
                            )
                            use_refiner = gr.Checkbox(
                                label="Use Refiner", value=True,
                                info="Apply refiner after generation ",
                                interactive=True
                            )
                        
                        generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Generated Image")
                        generated_image = gr.Image(
                            label="Generated Image",
                            format="png",
                            show_download_button=True,
                            type="pil",
                            height=600
                        )
                        generation_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Ready to generate"
                        )
            
            # Tab 2: Prompt Enhancement
            with gr.Tab("✨ Prompt Enhancement"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Prompt Enhancement Settings")
                        gr.Markdown("**Model**: HunyuanImage v2.1 Reprompt Model")
                        
                        # enhance_use_distilled = gr.Checkbox(
                        #     label="Use Distilled Model",
                        #     value=False,
                        #     info="For loading the reprompt model"
                        # )
                        enhance_use_distilled = False
                        
                        original_prompt = gr.Textbox(
                            label="Original Prompt",
                            placeholder="A cat sitting on a table",
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
                            value="Ready to enhance"
                        )
            
            # # Tab 3: Image Refinement
            # with gr.Tab("🔧 Image Refinement"):
            #     with gr.Row():
            #         with gr.Column(scale=1):
            #             gr.Markdown("### Refinement Settings")
            #             gr.Markdown("**Model**: HunyuanImage v2.1 Refiner")
                        
            #             input_image = gr.Image(
            #                 label="Input Image",
            #                 type="pil",
            #                 height=300
            #             )
                        
            #             refine_prompt = gr.Textbox(
            #                 label="Refinement Prompt",
            #                 placeholder="Make the image more detailed and high quality",
            #                 lines=2,
            #                 value="Make the image more detailed and high quality"
            #             )
                        
            #             refine_negative_prompt = gr.Textbox(
            #                 label="Negative Prompt",
            #                 placeholder="",
            #                 lines=2,
            #                 value=""
            #             )
                        
            #             with gr.Row():
            #                 refine_width = gr.Slider(
            #                     minimum=512, maximum=2048, step=64, value=2048,
            #                     label="Width", info="Output width"
            #                 )
            #                 refine_height = gr.Slider(
            #                     minimum=512, maximum=2048, step=64, value=2048,
            #                     label="Height", info="Output height"
            #                 )
                        
            #             with gr.Row():
            #                 refine_steps = gr.Slider(
            #                     minimum=1, maximum=20, step=1, value=4,
            #                     label="Refinement Steps", info="More steps = more refinement"
            #                 )
            #                 refine_guidance = gr.Slider(
            #                     minimum=1.0, maximum=10.0, step=0.1, value=3.5,
            #                     label="Guidance Scale", info="How strongly to follow the prompt"
            #                 )
                        
            #             refine_seed = gr.Number(
            #                 label="Seed", value=-1, precision=0,
            #                 info="Random seed for reproducibility"
            #             )
                        
            #             refine_btn = gr.Button("🔧 Refine Image", variant="primary", size="lg")
                    
            #         with gr.Column(scale=1):
            #             gr.Markdown("### Refined Image")
            #             refined_image = gr.Image(
            #                 label="Refined Image",
            #                 type="pil",
            #                 height=600
            #             )
            #             refinement_status = gr.Textbox(
            #                 label="Status",
            #                 interactive=False,
            #                 value="Ready to refine"
            #             )
        
        # Event handlers
        generate_btn.click(
            fn=app.generate_image,
            inputs=[
                prompt, negative_prompt, width, height, num_inference_steps,
                guidance_scale, seed, use_reprompt, use_refiner # , use_distilled
            ],
            outputs=[generated_image, generation_status]
        )
        
        enhance_btn.click(
            fn=app.enhance_prompt,
            inputs=[original_prompt],
            outputs=[enhanced_prompt, enhancement_status]
        )
        
        # refine_btn.click(
        #    fn=app.refine_image,
        #    inputs=[
        #        input_image, refine_prompt, refine_negative_prompt,
        #        refine_width, refine_height, refine_steps, refine_guidance, refine_seed
        #    ],
        #    outputs=[refined_image, refinement_status]
        # )
        
        # Additional info
        gr.Markdown(
            """
            ### 📝 Usage Tips
            
            **Text-to-Image Generation:**
            - Use descriptive prompts with specific details
            - Adjust guidance scale: higher values follow prompts more closely
            - More inference steps generally produce better quality
            - Enable reprompt for automatic prompt enhancement
            - Enable refiner for additional quality improvement
            
            **Prompt Enhancement:**
            - Enter your basic prompt idea
            - The AI will enhance it with better structure and details
            - Enhanced prompts often produce better results
            
            **Image Refinement:**
            - Upload any image you want to improve
            - Describe what improvements you want in the refinement prompt
            - The refiner will enhance details and quality
            - Works best with images generated by HunyuanImage
            """,
            elem_classes="model-info"
        )
    
    return demo

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch HunyuanImage Gradio App")
    parser.add_argument("--no-auto-load", action="store_true", help="Disable auto-loading pipeline on startup")
    parser.add_argument("--use-distilled", action="store_true", help="Use distilled model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--port", type=int, default=8081, help="Port to run the app on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    # Create and launch the interface
    auto_load = not args.no_auto_load
    demo = create_interface(auto_load=auto_load, use_distilled=args.use_distilled, device=args.device)
    
    print("🚀 Starting HunyuanImage Gradio App...")
    print(f"📱 The app will be available at: http://{args.host}:{args.port}")
    print(f"🔧 Auto-load pipeline: {'Yes' if auto_load else 'No'}")
    print(f"🎯 Model type: {'Distilled' if args.use_distilled else 'Non-distilled'}")
    print(f"💻 Device: {args.device}")
    print("⚠️  Make sure you have the required model checkpoints downloaded!")
    
    demo.launch(
        server_name=args.host,
        # server_port=args.port,
        share=False,
        show_error=True,
        quiet=False,
        max_threads=1,  # Default: sequential processing (recommended for GPU apps)
        # max_threads=4,  # Enable parallel processing (requires more GPU memory)
    )