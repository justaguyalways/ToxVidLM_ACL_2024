import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_one_epoch(model, train_dataloader, optimizer, config, devices=None):
    model = model.to(config.device)
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_dataloader):
        batch = {key: value.to(config.device) for key, value in batch.items()}
 
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["loss"]
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        total_loss += loss.item()
        

    average_loss = total_loss / len(train_dataloader)
    
    return average_loss
    

def validate(model, val_dataloader, config, devices=None):
    model = model.to(config.device)
    model.eval()
    task_metrics = {task: {"accuracy": 0.0, "f1": 0.0} for task in config.tasks}
    all_labels = {task: [] for task in config.tasks}
    all_predictions = {task: [] for task in config.tasks}

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            inputs = {key: value.to(config.device) for key, value in batch.items()}
            
            outputs = model(inputs)

            for task in config.tasks:
                # print(outputs[task], batch[task])

                task_predictions = torch.argmax(outputs[task], dim=1).detach().cpu().numpy()
                task_labels = torch.argmax(batch[task], dim=1).detach().cpu().numpy()

                task_accuracy = accuracy_score(task_labels, task_predictions)
                task_f1 = f1_score(task_labels, task_predictions, average="weighted")

                task_metrics[task]["accuracy"] += task_accuracy
                task_metrics[task]["f1"] += task_f1

                all_labels[task].extend(task_labels)
                all_predictions[task].extend(task_predictions)

    for task in config.tasks:
        task_metrics[task]["accuracy"] /= len(val_dataloader)
        task_metrics[task]["f1"] /= len(val_dataloader)
        task_metrics[task]["classification_report"] = classification_report(all_labels[task], all_predictions[task])
        print(task, " " , task_metrics[task]["accuracy"] , " ", task_metrics[task]["f1"], " ", task_metrics[task]["classification_report"], "\n")
        
    return task_metrics, all_labels, all_predictions

def train_model(model, train_dataloader, val_dataloader, config, num_epochs, track_task, track_metric, devices=None):
    
    model = model.to(config.device)
    history = {"train_validation": []}
    
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-5)
    
    best_val_metric = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} out of {num_epochs}")

        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, config)
        
        print(train_loss)

        # Validation
        val_metrics, _, _ = validate(model, val_dataloader, config)
        
        # print(val_metrics)

        # Save metrics to history
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_metrics": val_metrics
        }
        
        history["train_validation"].append(epoch_data)
        
        json_save_path = config.results_directory + config.file_name + ".json"

        # Save to JSON file
        with open(json_save_path, 'w') as json_file:
            json.dump(history, json_file)
        
        
        if val_metrics[track_task][track_metric] > best_val_metric:
            best_val_metric = val_metrics[track_task][track_metric]
            # torch.save(model.state_dict(), config.directory + config.file_name + ".pth")

    
    print("Training finished!")


