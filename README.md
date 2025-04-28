### This is the code refine version for fedora project

- Increase the readability
- Fix some bugs in data loader

### Note:

- for running experiment in Lab server, it is good to use parallelly_run.sh
- for cloud environment, it is good to use singly_run.sh
- **Please make sure commenting out the “USE_FP16="--fp16”. This is just a cuda memory optimization, but it appears that it will affect the accuracy, so DON’T USE IT!!!!!**
- For the gradient accumulation, it is ok to use.
