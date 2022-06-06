import torch
from torch import nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class SASCL_FeatureExtractor_Score(nn.Module):
    def __init__(
        self,
        rules_classifier,
        features_extracted_per_branch = 5,
        answers_count = 8,
        attr_branches = 10,
        total_rules = 4):

        super().__init__()

        self.rules_classifier = rules_classifier

        self.amswers_count = answers_count
        self.attr_branches = attr_branches
        self.total_rules = total_rules
        self.features_extracted_per_branch = features_extracted_per_branch

        for p in rules_classifier.parameters():
            p.requires_grad = False

        for i in range(attr_branches):
            branch = getattr(rules_classifier, "attr_branch%d" % i)

            branch.to_logit = Identity()

        self.to_logit = nn.Linear(answers_count * attr_branches * features_extracted_per_branch, answers_count)

    def forward(self, sets):
        rules_preds = self.rules_classifier(sets)

        rules_preds_per_set = torch.reshape(rules_preds, (-1, self.amswers_count, self.features_extracted_per_branch, self.attr_branches))

        probs = self.softmax(rules_preds_per_set)

        features = torch.flatten(probs, 1)
        
        logits = self.to_logit(features)
        return logits