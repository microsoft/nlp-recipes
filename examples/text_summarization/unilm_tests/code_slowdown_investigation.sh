## tests on hludsvmnc24rsv3, fine unilm-large-case for 100 steps
## output of nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.79       Driver Version: 410.79       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
## output of nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130

# fine-tune within s2s-ft docker container
# output from conda list
# pytorch                   1.2.0           py3.6_cuda10.0.130_cudnn7.6.2_0    pytorch
# cudatoolkit               10.0.130   
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
Iter (loss=0.559) lr=0.0000003: 100%|#####################################################################################################################################3| 199/200 [01:39<00:00,  2.10it/s]03/01/2020 21:11:36 - INFO - __main__ -   Saving model checkpoint 100
Iter (loss=0.559) lr=0.0000003: 100%|######################################################################################################################################| 200/200 [01:43<00:00,  1.53s/it]

# nlp_gpu, output from conda list
# torch                     1.2.0                    pypi_0    pypi
# cudatoolkit               9.2                           0  
Iteration: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:36<00:00,  2.08it/s]

# nlp_gpu_new, output from conda list
# pytorch                   1.2.0           cuda100py36h938c94c_0  
# cudatoolkit               10.0.130                      0
# python abstractive_summarization_unilm_cnndm.py --fp16 true
OMP: Info #154: KMP_AFFINITY: Initial OS proc set respected: 0-23
OMP: Info #213: KMP_AFFINITY: decoding x2APIC ids.
OMP: Info #276: KMP_AFFINITY: Affinity capable, using global cpuid leaf 11 info
OMP: Info #156: KMP_AFFINITY: 24 available OS procs
OMP: Info #157: KMP_AFFINITY: Uniform topology
OMP: Info #191: KMP_AFFINITY: 2 sockets x 12 cores/socket x 1 thread/core (24 total cores)
OMP: Info #215: KMP_AFFINITY: OS proc to physical thread map:
OMP: Info #171: KMP_AFFINITY: OS proc 0 maps to socket 0 core 0 
OMP: Info #171: KMP_AFFINITY: OS proc 1 maps to socket 0 core 1 
OMP: Info #171: KMP_AFFINITY: OS proc 2 maps to socket 0 core 2 
OMP: Info #171: KMP_AFFINITY: OS proc 3 maps to socket 0 core 3 
OMP: Info #171: KMP_AFFINITY: OS proc 4 maps to socket 0 core 4 
OMP: Info #171: KMP_AFFINITY: OS proc 5 maps to socket 0 core 5 
OMP: Info #171: KMP_AFFINITY: OS proc 6 maps to socket 0 core 6 
OMP: Info #171: KMP_AFFINITY: OS proc 7 maps to socket 0 core 7 
OMP: Info #171: KMP_AFFINITY: OS proc 8 maps to socket 0 core 8 
OMP: Info #171: KMP_AFFINITY: OS proc 9 maps to socket 0 core 9 
OMP: Info #171: KMP_AFFINITY: OS proc 10 maps to socket 0 core 10 
OMP: Info #171: KMP_AFFINITY: OS proc 11 maps to socket 0 core 11 
OMP: Info #171: KMP_AFFINITY: OS proc 12 maps to socket 1 core 0 
OMP: Info #171: KMP_AFFINITY: OS proc 13 maps to socket 1 core 1 
OMP: Info #171: KMP_AFFINITY: OS proc 14 maps to socket 1 core 2 
OMP: Info #171: KMP_AFFINITY: OS proc 15 maps to socket 1 core 3 
OMP: Info #171: KMP_AFFINITY: OS proc 16 maps to socket 1 core 4 
OMP: Info #171: KMP_AFFINITY: OS proc 17 maps to socket 1 core 5 
OMP: Info #171: KMP_AFFINITY: OS proc 18 maps to socket 1 core 6 
OMP: Info #171: KMP_AFFINITY: OS proc 19 maps to socket 1 core 7 
OMP: Info #171: KMP_AFFINITY: OS proc 20 maps to socket 1 core 8 
OMP: Info #171: KMP_AFFINITY: OS proc 21 maps to socket 1 core 9 
OMP: Info #171: KMP_AFFINITY: OS proc 22 maps to socket 1 core 10 
OMP: Info #171: KMP_AFFINITY: OS proc 23 maps to socket 1 core 11 
OMP: Info #251: KMP_AFFINITY: pid 26368 tid 26368 thread 0 bound to OS proc set 0 
Iteration: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [03:01<00:00,  1.06it/s]


# python -m torch.distributed.launch --nproc_per_node=4 --nnode=1 abstractive_summarization_unilm_cnndm_torchdist.py --fp16 true
Iteration: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [01:41<00:00,  2.05it/s]