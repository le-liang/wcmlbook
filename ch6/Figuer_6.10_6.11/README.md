# A code implementation of Soft_DSGD

This code is a copy from the reference code link for Figure 6.10 & 6.11.

The reference code link: [here](https://github.com/haoyye/Unrealiable_Decentralized_FL)

## Run experiments
Decentralized training with retransmittion (baseline).
~~~
python main.py --communication_mode TCP --num_workers 16
~~~
Decentralized training with Soft-DSGD, uniform weights (proposed)
~~~
python main.py --communication_mode UDP --num_workers 16 --weights_type UNI
~~~
Decentralized training with Soft-DSGD, optimized weights (proposed)
~~~
python main.py --communication_mode TCP --num_workers 16 --weights_type OPT
~~~
