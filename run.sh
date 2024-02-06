
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
echo $free_mem

while [ $free_mem -lt 10000 ]; do
  free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
  echo $free_mem
  sleep 10
done

python main.py --multirun model.arch=CNN,RNN

