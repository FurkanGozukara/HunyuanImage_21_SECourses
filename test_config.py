#!/usr/bin/env python
"""Test configuration saving and loading with all parameters."""

import json
from pathlib import Path
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add the current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from secourses_app import ConfigManager, CONFIGS_DIR

def test_config_system():
    """Test that all parameters are saved and loaded correctly."""
    
    # Initialize config manager
    config_mgr = ConfigManager(CONFIGS_DIR)
    
    # Define test parameters with all fields
    test_params = {
        'model_type': 'distilled',
        'enable_dit_offloading': False,
        'enable_reprompt_offloading': False,
        'enable_refiner_offloading': True,
        'prompt': 'A beautiful landscape with mountains',
        'negative_prompt': 'blurry, low quality',
        'aspect_ratio': '16:9',
        'width': 1920,
        'height': 1080,
        'num_inference_steps': 75,
        'guidance_scale': 5.0,
        'seed': 42,
        'use_reprompt': True,
        'use_refiner': True,
        'refiner_steps': 8,
        'auto_enhance': True,
        'multi_line_prompt': True,
        'num_generations': 3,
        'main_shift': 6,
        'refiner_shift': 2,
        'refiner_guidance': 2.0
    }
    
    # Test saving configuration
    print("Testing configuration save...")
    config_name = "test_config"
    success = config_mgr.save_config(config_name, test_params)
    assert success, "Failed to save configuration"
    print(f"[OK] Configuration '{config_name}' saved successfully")
    
    # Test loading configuration
    print("\nTesting configuration load...")
    loaded_params = config_mgr.load_config(config_name)
    assert loaded_params is not None, "Failed to load configuration"
    print(f"[OK] Configuration '{config_name}' loaded successfully")
    
    # Verify all parameters match
    print("\nVerifying all parameters...")
    for key, value in test_params.items():
        assert key in loaded_params, f"Missing parameter: {key}"
        assert loaded_params[key] == value, f"Parameter mismatch for {key}: expected {value}, got {loaded_params[key]}"
        print(f"  [OK] {key}: {value}")
    
    # Test listing configurations
    print("\nTesting configuration listing...")
    configs = config_mgr.list_configs()
    assert config_name in configs, f"Configuration '{config_name}' not in list"
    print(f"[OK] Found {len(configs)} configuration(s): {', '.join(configs)}")
    
    # Test last config
    print("\nTesting last configuration...")
    last_config = config_mgr.load_last_config()
    assert last_config is not None, "Failed to load last configuration"
    assert last_config == test_params, "Last configuration doesn't match saved parameters"
    print("[OK] Last configuration matches saved parameters")
    
    # Clean up test config
    config_path = CONFIGS_DIR / f"{config_name}.json"
    if config_path.exists():
        config_path.unlink()
        print(f"\n[OK] Cleaned up test configuration file")
    
    print("\n[SUCCESS] All configuration tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_config_system()
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)