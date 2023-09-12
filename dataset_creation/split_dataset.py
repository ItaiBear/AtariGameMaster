import minari

dataset = minari.load_dataset("TetrisA-v0-itai-v2")

split_datasets = minari.split_dataset(dataset, sizes=[1,1])

dataset1 = split_datasets[0]
dataset2 = split_datasets[1]

if dataset1.total_steps > dataset2.total_steps:
    dataset = dataset1
else:
    dataset = dataset2
    
print(dataset.total_steps)