$Env:IS_OFFLINE=$true
python .\src\rp_handler.py --model_name "cosxledit" --test_input "D:\OneDrive\Projects\photobooth\worker-sdxl\test_input.json" --rp_serve_api --rp_api_host "0.0.0.0"
