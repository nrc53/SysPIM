# SysPIM
Framework for performance evaluation of video analytics workloads on emerging processing-in-memory architectures (Work in Progress)

**Sample Command**

python main.py --pipeline detect_track --pim_accelerator crossbar --accelerator_config crossbar1 --report_prefix detect_track --report_dir workdir


yaml files describing the sample video analytics pipelines are in the directory "pipelines"

yaml files with neural network descriptions are in the directory "neural_networks"

yaml files with architecture parameters are in the directory "arch_parameters"

**Please refer to below paper for details about SysPIM framework**

N. Challapalle and V. Narayanan, "Performance Evaluation of Video Analytics Workloads on Emerging Processing-In-Memory Architectures," 2022 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), 2022, pp. 158-163, doi: 10.1109/ISVLSI54635.2022.00040.
