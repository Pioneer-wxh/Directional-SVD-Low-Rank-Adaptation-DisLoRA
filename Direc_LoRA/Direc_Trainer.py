from transformers import Trainer, TrainingArguments
import torch

class Direc_TrainingArguments(TrainingArguments):
    def __init__(self, ortho_lambda=4e-4, **kwargs):
        super().__init__(**kwargs)
        self.ortho_lambda = ortho_lambda

class Direc_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # First call parent class compute_loss to get original task loss
        task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # Calculate orthogonal constraint loss
        ortho_loss = self.calc_ortho(model)
        
        if ortho_loss is not None:
            if int(self.args.ortho_lambda) == 1:
                # Pareto optimization implementation
                # Use a dynamic weight adjustment method to balance two losses
                # Calculate relative magnitude of two losses
                ratio = ortho_loss.detach() / (task_loss.detach() + 1e-8)  # Avoid division by zero
                
                # Use softmax-like method to calculate dynamic weights
                alpha_task = torch.exp(-ratio) / (torch.exp(-ratio) + torch.exp(-1/ratio + 1e-8))
                alpha_ortho = 1.0 - alpha_task
                
                # Calculate total loss after Pareto optimization
                total_loss = alpha_task * task_loss + alpha_ortho * ortho_loss
            else:
                # Non-Pareto optimization case, maintain original weighted method
                total_loss = task_loss + self.args.ortho_lambda * ortho_loss
        else:
            total_loss = task_loss
            
        return (total_loss, outputs) if return_outputs else total_loss

    
    def calc_ortho(self, model):
        ortho_loss = 0.0
        den = 0
        for name, param in model.named_parameters():
            if "Direc_Ur" in name:
                u = param
                iu = torch.eye(u.shape[1], device=u.device)
                iu.requires_grad = False
                u_loss = torch.norm(u.T @ u - iu, p="fro")
                ortho_loss += u_loss
                den += 1
            if "Direc_Vhr" in name:
                vh = param
                ivh = torch.eye(vh.shape[0], device=vh.device)
                ivh.requires_grad = False
                vh_loss = torch.norm(vh @ vh.T - ivh, p="fro")
                ortho_loss += vh_loss
                den += 1
        if den != 0:
            return ortho_loss / den
        else:
            return None