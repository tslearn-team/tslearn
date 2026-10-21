CERT=$1

PWD=$SLICES_PWD

PROJECT_NAME='p16-programia'
TSLEARN_REPO_URL='https://github.com/tslearn-team/tslearn.git'


JOB_ID=$(gpulab-cli --cert $CERT -p $PWD submit --project=$PROJECT_NAME <<EOF
{
    "name": "Test GPU",
    "description": "Test GPU",
    "request": {
        "resources": {
            "cpus": 1,
            "gpus": 1,
            "cpuMemoryGb": 2,
            "minCudaVersion": 12
        },
        "docker": {
            "image": "gitlab.ilabt.imec.be:4567/ilabt/gpu-docker-stacks/pytorch-notebook:cuda12.6-latest",
            "command": "bash -c 'git clone $TSLEARN_REPO_URL tslearn; cd tslearn; git reset --hard $GITHUB_SHA; python -m pip install --no-cache-dir .[tests,all_features]; pytest -v'"
        }
    }
}
EOF
)
echo $JOB_ID