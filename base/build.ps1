param([string]$Name="jumpmyman/sdxl_gen",$Version="latest",$BaseVersion="0.4.2",[switch]$Cache, [switch]$Loader,[switch]$Base,
    [switch]$NoTurbo,
    [switch]$NoBase,
    [switch]$NoRefiner,
    [switch]$NoCanny,
    [switch]$Worker,
    [string]$PipelineName)
pushd "$PSScriptRoot"
try {
    if($Cache) {
        docker image build --build-arg "BASE_VERSION=$BaseVersion" -t ${Name}:cached_models -t ${Name}:cached_models -f "./cached_models/Dockerfile" "./cached_models"
    }
    if($Base) {
        docker image build --target runpod_base --build-arg "BASE_VERSION=$BaseVersion" -t ${Name}:base -t ${Name}:base -t ${Name}:${BaseVersion} -f "./Dockerfile" "../"
    }
    $ModelsLabel="loaded_models"
    $BaseModel = $NoBase ? "" : "base";
    $BaseModel += $NoTurbo ? "" : "_turbo";
    $BaseModel.TrimStart("_")
    $ModelsLabel += $NoBase ? "_none" : "_$BaseModel";
    $ModelsLabel += $NoRefiner ? "" : "_refiner";
    $ModelsLabel += $NoCanny ? "" : "_canny";
    if($Loader) {
        # ARG REVISION=true
        # #base or turbo
        # ARG BASE_MODEL=base
        # #true or false
        # ARG REFINER=true
        # # base, lora or false
        # ARG CANNY=base
        $BuildArgs = @("--build-arg","BASE_MODEL=$BaseModel", 
            "--build-arg","REFINER=$($NoRefiner ? "false" : "true")",
            "--build-arg", "CANNY=$($NoCanny ? "false" : "base")"
            );
        docker image build @BuildArgs  -t ${Name}:$ModelsLabel -t ${Name}:loaded_models-latest --target "model_loader_final" -f "./loaded_models.Dockerfile" "../"
        # docker image build --build-arg "BASE_MODEL=turbo" --build-arg "REFINER=false" --build-arg "CANNY=base" -t ${Name}:loaded_models_turbo --target "sdxl_loader_final" -f "./loaded_models.Dockerfile" "../"
        # docker image build --build-arg "BASE_MODEL=base" --build-arg "REFINER=false" --build-arg "CANNY=none" -t ${Name}:loaded_models_mid --target "sdxl_loader_final" -f "./loaded_models.Dockerfile" "../"
    }
    if($Worker) {
        $BuildArgs = @("--build-arg","BASE_LABEL=$BaseVersion", "--build-arg","MODELS_LABEL=$ModelsLabel");
        if($ModelsLabel -eq "loaded_models_none") {
            $BuildArgs += @('--build-arg','NO_MODELS=true')
        }
        if($PipelineName) {
            $BuildArgs += @("--build-arg","MODEL_NAME=$PipelineName")
        }
        docker image build @BuildArgs -t ${Name}:${BaseVersion}-${Version} -t ${Name}:latest -t ${Name}:${Version} "../"
    }
}
finally {
    popd
}