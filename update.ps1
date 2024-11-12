param($Container)
if(!$Container) {
	$Container=$Global:Container
}
echo "docker container cp $PSScriptRoot/src/*.py $Container`:/"
docker container cp "$PSScriptRoot/src/*.py" "$Container"`:"/"
docker container restart "$Container" 
docker container logs -f "$Container"
