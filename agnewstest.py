from openprompt.data_utils.text_classification_dataset import AgnewsProcessor  # load specific news processor
from openprompt.data_utils.data_sampler import FewShotSampler # few shot 
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import SoftVerbalizer
from openprompt import PromptForClassification
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score


dataset = {}
# trainvalid_dataset = processor.get_train_examples(dataset_path)
dataset['train'] = AgnewsProcessor().get_train_examples("/root/gdrive/MyDrive/OpenPrompt/datasets/TextClassification/agnews")
sampler  = FewShotSampler(num_examples_per_label=20, num_examples_per_label_dev=20, also_sample_dev=True)
dataset['train'], dataset['validation'] = sampler(dataset['train'])
dataset['test'] = AgnewsProcessor().get_test_examples("/root/gdrive/MyDrive/OpenPrompt/datasets/TextClassification/agnews")
print(len(dataset['train']))
print(dataset['train'][0])

plm, tokenizer, model_config, WrapperClass=load_plm("bert", "bert-base-cased") # load pretrained model
# define the template
template = ManualTemplate(tokenizer=tokenizer, text='The headline is {"placeholder":"text_a"} and the content is {"placeholder":"text_b"} In this news, the topic is {"mask"}.')
verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=4)
wrapped_example = template.wrap_one_example(dataset['train'][0]) 
#print('wrapped_example:',wrapped_example)

# in the pipline base 
# maximum 512 tokens for bert, tail truncation, not fill the mask with target text
train_dataloader = PromptDataLoader(dataset=dataset['train'], template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, batch_size=4, teacher_forcing=False, predict_eos_token=False, 
    truncate_method='tail')

val_dataloader = PromptDataLoader(dataset=dataset['validation'], template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, batch_size=4, teacher_forcing=False, predict_eos_token=False, 
    truncate_method='tail')

test_dataloader = PromptDataLoader(dataset=dataset['test'], template=template, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, batch_size=4, teacher_forcing=False, predict_eos_token=False, 
    truncate_method='tail')

# build the prompt model
prompt_model = PromptForClassification(plm=plm,template=template, verbalizer=verbalizer, freeze_plm=False)

use_cuda = torch.cuda.is_available()
if use_cuda:
    prompt_model=  prompt_model.cuda()
#prompt_model=  prompt_model.cuda()
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']

## except biases and Layer Normalization weights with a weight decay of 0.01 and  includes biases and Layer Normalization weights with no weight decay 
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# tow different learning rate
optimizer_grouped_parameters2 = [
    {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
    {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
]


optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
optimizer2 = AdamW(optimizer_grouped_parameters2)

def accuracy(preds, labels):
    preds_class = torch.argmax(preds, dim=1)
    return (preds_class == labels).float().mean()


def evaluate_model(model, dataloader, loss_func, use_cuda):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_steps = 0

    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            total_steps += 1
            if use_cuda:
                inputs = inputs.cuda()

            logits = model(inputs)
            labels = inputs['label']

            loss = loss_func(logits, labels)
            total_loss += loss.item()

            acc = accuracy(logits, labels)
            total_acc += acc.item()

    return total_loss / total_steps, total_acc / total_steps


# Start training
for epoch in range(5):
    tot_loss = 0 
    tot_acc = 0  
    num_steps = 0  
    for step, inputs in enumerate(train_dataloader):
        num_steps += 1
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        optimizer2.step()
        optimizer2.zero_grad()

        acc = accuracy(logits, labels)
        tot_acc += acc.item() 

        print(f"Epoch: {epoch+1}, Step: {step+1}, Loss: {tot_loss/(step+1)}, Accuracy: {tot_acc/(step+1)}")

    print(f"\nAverage loss for Epoch {epoch+1}: {tot_loss/num_steps}")
    print(f"Average accuracy for Epoch {epoch+1}: {tot_acc/num_steps}\n")    

    val_loss, val_acc = evaluate_model(prompt_model, val_dataloader, loss_func, use_cuda)
    print(f"Validation loss for Epoch {epoch+1}: {val_loss}")
    print(f"Validation accuracy for Epoch {epoch+1}: {val_acc}\n")

    test_loss, test_acc = evaluate_model(prompt_model, test_dataloader, loss_func, use_cuda)
    print(f"Test loss for Epoch {epoch+1}: {test_loss}")
    print(f"Test accuracy for Epoch {epoch+1}: {test_acc}\n")




