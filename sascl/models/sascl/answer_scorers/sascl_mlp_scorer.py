import torch
from torch import nn

class SASCL_MLP_Score(nn.Module):
    def __init__(
        self,
        rules_classifier,
        answers_count = 8,
        attr_branches = 10,
        total_rules = 4):

        super().__init__()

        self.rules_classifier = rules_classifier

        self.amswers_count = answers_count
        self.attr_branches = attr_branches
        self.total_rules = total_rules

        for p in rules_classifier.parameters():
            p.requires_grad = False

        self.softmax = nn.Softmax(dim=2)

        self.to_logit = nn.Linear(answers_count * attr_branches * (total_rules + 1), answers_count)

    def forward(self, sets):
        rules_preds = self.rules_classifier(sets)

        rules_preds_per_set = torch.reshape(rules_preds, (-1, self.amswers_count, self.total_rules + 1, self.attr_branches))

        probs = self.softmax(rules_preds_per_set)

        features = torch.flatten(probs, 1)
        
        logits = self.to_logit(features)
        return logits