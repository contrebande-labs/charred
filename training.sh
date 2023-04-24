export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=34359738368
#export JAX_PLATFORMS=''
python3 main.py --push_to_hub --max_train_steps 64 --num_train_epochs 3