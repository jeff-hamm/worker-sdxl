pushd "$PSScriptRoot"
try {
    docker image build --target runpod_base -t runpod_api_base:0.4.2-cuda11.8.0 -t runpod_api_base:latest "../"
    docker image build -t runpod_sdxl_api_base:0.4.2-cuda11.8.0 -t runpod_sdxl_api_base:latest "../"
}
finally {
    popd
}