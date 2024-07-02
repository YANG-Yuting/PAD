from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AutoConfig
from config import args
from torch.nn import CrossEntropyLoss, MSELoss
import torch
import transformers


def eval_model(model, dataloader_test):  # inputs_x is list of str
    model.eval()
    correct = 0.0
    with torch.no_grad():
        all_logits, all_labels = [], []
        for idx, (*x, y) in enumerate(dataloader_test):
            input_ids, input_mask, segment_ids, labels = x[0].cuda(args.rank), x[1].cuda(args.rank), x[2].cuda(args.rank), y.cuda(args.rank)
            if torch.cuda.device_count() > 1:
                model.module.inference = True
                if args.target_model == 'bart':
                    outputs = model.module(input_ids=input_ids, attention_mask=input_mask)
                else:
                    outputs = model.module(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            else:
                model.inference = True
                if args.target_model == 'bart':
                    outputs = model(input_ids=input_ids, attention_mask=input_mask)
                else:
                    outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = outputs.logits
            all_logits.append(logits)
            all_labels.append(labels)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        preds = torch.argmax(all_logits, dim=1)

        data_size = preds.shape[0]
        correct += torch.sum(torch.eq(preds, all_labels))
        acc = (correct.cpu().numpy()) / float(data_size)
    return acc


"""重载自BertForSequenceClassification：用以模型集成的训练和推理"""
class EnsembleBERT(BertForSequenceClassification):
    def __init__(self, config):
        super(EnsembleBERT, self).__init__(config)
        self.logits_ensemble = 'vote'
        self.inference = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        all_loss, all_logits, all_hidden_states, all_attentions, aux_outs = [], [], [], [], []
        for i in range(args.num_models):

            # from original BertForSequenceClassification
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                return_dict=return_dict, model_id=i)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            if labels is not None:
                if args.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
                if len(loss.size()) == 0:  # tiz
                    loss = torch.unsqueeze(loss, 0)  # tensor(1) -> tensor([1])
                all_loss.append(loss)   # tiz
            all_logits.append(logits)
            all_hidden_states.append(outputs.hidden_states)  # tuple: 13 * (b_size, seq_len, hidden_size)
            all_attentions.append(outputs.attentions)  # None

            if args.perturb_attention:
                # aux_outs.append(torch.reshape(self.bert.encoder.layer[0].attention.self.params_aux, [input_ids.shape[0], -1]))
                aux_outs.append(self.bert.encoder.layer[args.modif_att_layer].attention.self.params_aux)  # [1,12,128,128]

        if self.inference:
            # 测试时，计算多个子模型的集成结果: loss, hidden_states, attentions取多个模型的平均；logits 取平均/投票
            esb_loss, esb_hidden_states, esb_attentions = None, None, None
            if labels is not None:
                all_loss = torch.cat(all_loss, dim=0)
                esb_loss = torch.mean(all_loss, dim=-1)
            if output_hidden_states:
                esb_hidden_states = []
                for layer_no in range(len(all_hidden_states[0])):
                    layer_hidden_states = torch.cat([all_hidden_states[i][layer_no].unsqueeze(0) for i in range(args.num_models)], dim=0)
                    esb_layer_hidden_states = torch.mean(layer_hidden_states, dim=0)
                    esb_hidden_states.append(esb_layer_hidden_states)
            if output_attentions:
                # 多个模型的attention求平均
                esb_attentions = []
                for layer_no in range(len(all_attentions[0])):
                    layer_attentions = torch.cat([all_attentions[i][layer_no].unsqueeze(0) for i in range(args.num_models)], dim=0)
                    esb_layer_attentions = torch.mean(layer_attentions, dim=0)
                    esb_attentions.append(esb_layer_attentions)
                # 直接返回多个attention
                # esb_attentions = all_attentions

            if self.logits_ensemble == 'avg':
                all_logits = torch.cat([al.unsqueeze(0) for al in all_logits], dim=0)
                esb_logits = torch.mean(all_logits, dim=0)
            elif self.logits_ensemble == 'vote':
                all_logits = torch.cat(all_logits, dim=1)  # 注意这里和训练阶段不同，是竖着拼接 [batch_size, num_classes*num_models]
                all_logits = all_logits.view(input_ids.size()[0], args.num_models, args.num_labels)  # batch_size, num_classes, num_models
                probs_boost = []
                for l in range(self.num_labels):
                    num = torch.sum(torch.eq(torch.argmax(all_logits, dim=2), l), dim=1)  # 获得几个集成模型的预测标签的比例作为对应标签的概率
                    prob = num.float() / float(args.num_models)
                    probs_boost.append(prob.view(input_ids.size()[0], 1))
                esb_logits = torch.cat(probs_boost, dim=1)

            return transformers.modeling_outputs.SequenceClassifierOutput(
                loss=esb_loss,
                logits=esb_logits,
                hidden_states=esb_hidden_states,
                attentions=esb_attentions,
            )
        else:
            # 训练时，返回多个子模型的预测结果
            if args.perturb_attention:
                # aux_outs: 是包含num_models个tensor的list，每个tensor [b_size, num_heads, seq_len, seq_len]
                return aux_outs, transformers.modeling_outputs.SequenceClassifierOutput(loss=all_loss, logits=all_logits, hidden_states=all_hidden_states, attentions=all_attentions)
            else:
                return transformers.modeling_outputs.SequenceClassifierOutput(loss=all_loss, logits=all_logits, hidden_states=all_hidden_states, attentions=all_attentions)


"""多个bert叠加"""
class MultipleBERT(torch.nn.Module):
    def __init__(self):
        super(MultipleBERT, self).__init__()
        self.logits_ensemble = 'vote'
        self.inference = False
        self.config = AutoConfig.from_pretrained('bert-base-uncased')
        if not self.inference:
            self.model0 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels, output_attentions=True).cuda()
            self.model1 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels, output_attentions=True).cuda()
            self.model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.num_labels, output_attentions=True).cuda()
            self.models = [self.model0, self.model1, self.model2]

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        all_loss, all_logits, all_hidden_states, all_attentions, aux_outs = [], [], [], [], []
        for i in range(args.num_models):
            # from original BertForSequenceClassification
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.models[i](input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                return_dict=return_dict)
            logits = outputs.logits
            all_logits.append(logits)

        if self.inference:
            # 测试时，计算多个子模型的集成结果: loss, hidden_states, attentions取多个模型的平均；logits 取平均/投票
            esb_loss, esb_hidden_states, esb_attentions = None, None, None
            if labels is not None:
                all_loss = torch.cat(all_loss, dim=0)
                esb_loss = torch.mean(all_loss, dim=-1)
            if output_hidden_states:
                esb_hidden_states = []
                for layer_no in range(len(all_hidden_states[0])):
                    layer_hidden_states = torch.cat([all_hidden_states[i][layer_no].unsqueeze(0) for i in range(args.num_models)], dim=0)
                    esb_layer_hidden_states = torch.mean(layer_hidden_states, dim=0)
                    esb_hidden_states.append(esb_layer_hidden_states)
            if output_attentions:
                # 多个模型的attention求平均
                esb_attentions = []
                for layer_no in range(len(all_attentions[0])):
                    layer_attentions = torch.cat([all_attentions[i][layer_no].unsqueeze(0) for i in range(args.num_models)], dim=0)
                    esb_layer_attentions = torch.mean(layer_attentions, dim=0)
                    esb_attentions.append(esb_layer_attentions)
                # 直接返回多个attention
                # esb_attentions = all_attentions

            if self.logits_ensemble == 'avg':
                all_logits = torch.cat([al.unsqueeze(0) for al in all_logits], dim=0)
                esb_logits = torch.mean(all_logits, dim=0)
            elif self.logits_ensemble == 'vote':
                all_logits = torch.cat(all_logits, dim=1)  # 注意这里和训练阶段不同，是竖着拼接 [batch_size, num_classes*num_models]
                all_logits = all_logits.view(input_ids.size()[0], args.num_models, args.num_labels)  # batch_size, num_classes, num_models
                probs_boost = []
                for l in range(args.num_labels):
                    num = torch.sum(torch.eq(torch.argmax(all_logits, dim=2), l), dim=1)  # 获得几个集成模型的预测标签的比例作为对应标签的概率
                    prob = num.float() / float(args.num_models)
                    probs_boost.append(prob.view(input_ids.size()[0], 1))
                esb_logits = torch.cat(probs_boost, dim=1)

            return transformers.modeling_outputs.SequenceClassifierOutput(
                loss=esb_loss,
                logits=esb_logits,
                hidden_states=esb_hidden_states,
                attentions=esb_attentions,
            )
        else:
            # 训练时，返回多个子模型的预测结果
            return transformers.modeling_outputs.SequenceClassifierOutput(loss=all_loss, logits=all_logits, hidden_states=all_hidden_states, attentions=all_attentions)


"""重载自BertForSequenceClassification：用以模型集成的训练和推理"""
# class EnsembleROBERTA(RobertaForSequenceClassification):
#     def __init__(self, config):
#         super(EnsembleROBERTA, self).__init__(config)
#         self.logits_ensemble = 'vote'
#         self.inference = True
#
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
#                 inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
#
#         all_loss, all_logits, all_hidden_states, all_attentions, aux_outs = [], [], [], [], []
#         for i in range(args.num_models):
#
#             # from original BertForSequenceClassification
#             return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#             outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                                 position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
#                                 output_attentions=output_attentions, output_hidden_states=output_hidden_states,
#                                 return_dict=return_dict, model_id=i)
#             sequence_output = outputs[0]
#             logits = self.classifier(sequence_output)
#             if labels is not None:
#                 if args.num_labels == 1:
#                     #  We are doing regression
#                     loss_fct = MSELoss()
#                     loss = loss_fct(logits.view(-1), labels.view(-1))
#                 else:
#                     loss_fct = CrossEntropyLoss()
#                     loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
#                 if len(loss.size()) == 0:  # tiz
#                     loss = torch.unsqueeze(loss, 0)  # tensor(1) -> tensor([1])
#                 all_loss.append(loss)   # tiz
#             all_logits.append(logits)
#             all_hidden_states.append(outputs.hidden_states)  # tuple: 13 * (b_size, seq_len, hidden_size)
#             all_attentions.append(outputs.attentions)  # None
#
#             if args.perturb_attention:
#                 # aux_outs.append(torch.reshape(self.bert.encoder.layer[0].attention.self.params_aux, [input_ids.shape[0], -1]))
#                 aux_outs.append(self.roberta.encoder.layer[args.modif_att_layer].attention.self.params_aux)  # [1,12,128,128]
#
#         if self.inference:
#             # 测试时，计算多个子模型的集成结果: loss, hidden_states, attentions取多个模型的平均；logits 取平均/投票
#             esb_loss, esb_hidden_states, esb_attentions = None, None, None
#             if labels is not None:
#                 all_loss = torch.cat(all_loss, dim=0)
#                 esb_loss = torch.mean(all_loss, dim=-1)
#             if output_hidden_states:
#                 esb_hidden_states = []
#                 for layer_no in range(len(all_hidden_states[0])):
#                     layer_hidden_states = torch.cat([all_hidden_states[i][layer_no].unsqueeze(0) for i in range(args.num_models)], dim=0)
#                     esb_layer_hidden_states = torch.mean(layer_hidden_states, dim=0)
#                     esb_hidden_states.append(esb_layer_hidden_states)
#             if output_attentions:
#                 # 多个模型的attention求平均
#                 esb_attentions = []
#                 for layer_no in range(len(all_attentions[0])):
#                     layer_attentions = torch.cat([all_attentions[i][layer_no].unsqueeze(0) for i in range(args.num_models)], dim=0)
#                     esb_layer_attentions = torch.mean(layer_attentions, dim=0)
#                     esb_attentions.append(esb_layer_attentions)
#                 # 直接返回多个attention
#                 # esb_attentions = all_attentions
#
#             if self.logits_ensemble == 'avg':
#                 all_logits = torch.cat([al.unsqueeze(0) for al in all_logits], dim=0)
#                 esb_logits = torch.mean(all_logits, dim=0)
#             elif self.logits_ensemble == 'vote':
#                 all_logits = torch.cat(all_logits, dim=1)  # 注意这里和训练阶段不同，是竖着拼接 [batch_size, num_classes*num_models]
#                 all_logits = all_logits.view(input_ids.size()[0], args.num_models, args.num_labels)  # batch_size, num_classes, num_models
#                 probs_boost = []
#                 for l in range(self.num_labels):
#                     num = torch.sum(torch.eq(torch.argmax(all_logits, dim=2), l), dim=1)  # 获得几个集成模型的预测标签的比例作为对应标签的概率
#                     prob = num.float() / float(args.num_models)
#                     probs_boost.append(prob.view(input_ids.size()[0], 1))
#                 esb_logits = torch.cat(probs_boost, dim=1)
#
#             return transformers.modeling_outputs.SequenceClassifierOutput(
#                 loss=esb_loss,
#                 logits=esb_logits,
#                 hidden_states=esb_hidden_states,
#                 attentions=esb_attentions,
#             )
#         else:
#             # 训练时，返回多个子模型的预测结果
#             if args.perturb_attention:
#                 # aux_outs: 是包含num_models个tensor的list，每个tensor [b_size, num_heads, seq_len, seq_len]
#                 return aux_outs, transformers.modeling_outputs.SequenceClassifierOutput(loss=all_loss, logits=all_logits, hidden_states=all_hidden_states, attentions=all_attentions)
#             else:
#                 return transformers.modeling_outputs.SequenceClassifierOutput(loss=all_loss, logits=all_logits, hidden_states=all_hidden_states, attentions=all_attentions)


class EnsembleROBERTA(RobertaForSequenceClassification):
    def __init__(self, config):
        super(EnsembleROBERTA, self).__init__(config)
        self.logits_ensemble = 'vote'
        self.inference = True
        # base_config = AutoConfig.from_pretrained('/pub/data/huangpei/PAT-AAAI23/TextFooler/models/roberta/mr_new')
        # base_config.perturb_attention = False
        # base_config.num_models = 1
        # self.base = RobertaForSequenceClassification.from_pretrained('/pub/data/huangpei/PAT-AAAI23/TextFooler/models/roberta/mr_new', config=base_config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):

        all_loss, all_logits, all_hidden_states, all_attentions, aux_outs = [], [], [], [], []
        for i in range(args.num_models):

            # from original BertForSequenceClassification
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
                                output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                return_dict=return_dict, model_id=i)
            sequence_output = outputs[0]
            logits = self.classifier(sequence_output)
            if labels is not None:
                if args.num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, args.num_labels), labels.view(-1))
                if len(loss.size()) == 0:  # tiz
                    loss = torch.unsqueeze(loss, 0)  # tensor(1) -> tensor([1])
                all_loss.append(loss)   # tiz
            all_logits.append(logits)
            all_hidden_states.append(outputs.hidden_states)  # tuple: 13 * (b_size, seq_len, hidden_size)
            all_attentions.append(outputs.attentions)  # None

            if args.perturb_attention:
                # aux_outs.append(torch.reshape(self.bert.encoder.layer[0].attention.self.params_aux, [input_ids.shape[0], -1]))
                aux_outs.append(self.roberta.encoder.layer[args.modif_att_layer].attention.self.params_aux)  # [1,12,128,128]
        if self.inference:
            # base_outputs = self.base(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            #                         position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
            #                         output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            #                         return_dict=return_dict)
            # all_logits.append(base_outputs.logits)
            # all_hidden_states.append(base_outputs.hidden_states)
            # all_attentions.append(base_outputs.attentions)  # None

            n_model = len(all_logits)

            # 测试时，计算多个子模型的集成结果: loss, hidden_states, attentions取多个模型的平均；logits 取平均/投票
            esb_loss, esb_hidden_states, esb_attentions = None, None, None
            if labels is not None:
                all_loss = torch.cat(all_loss, dim=0)
                esb_loss = torch.mean(all_loss, dim=-1)
            if output_hidden_states:
                esb_hidden_states = []
                for layer_no in range(len(all_hidden_states[0])):
                    layer_hidden_states = torch.cat([all_hidden_states[i][layer_no].unsqueeze(0) for i in range(n_model)], dim=0)
                    esb_layer_hidden_states = torch.mean(layer_hidden_states, dim=0)
                    esb_hidden_states.append(esb_layer_hidden_states)
            if output_attentions:
                # 多个模型的attention求平均
                esb_attentions = []
                for layer_no in range(len(all_attentions[0])):
                    layer_attentions = torch.cat([all_attentions[i][layer_no].unsqueeze(0) for i in range(n_model)], dim=0)
                    esb_layer_attentions = torch.mean(layer_attentions, dim=0)
                    esb_attentions.append(esb_layer_attentions)
                # 直接返回多个attention
                # esb_attentions = all_attentions

            if self.logits_ensemble == 'avg':
                all_logits = torch.cat([al.unsqueeze(0) for al in all_logits], dim=0)
                esb_logits = torch.mean(all_logits, dim=0)
            elif self.logits_ensemble == 'vote':
                all_logits = torch.cat(all_logits, dim=1)  # 注意这里和训练阶段不同，是竖着拼接 [batch_size, num_classes*num_models]
                all_logits = all_logits.view(input_ids.size()[0], n_model, args.num_labels)  # batch_size, num_classes, num_models
                probs_boost = []
                for l in range(self.num_labels):
                    num = torch.sum(torch.eq(torch.argmax(all_logits, dim=2), l), dim=1)  # 获得几个集成模型的预测标签的比例作为对应标签的概率
                    prob = num.float() / float(n_model)
                    probs_boost.append(prob.view(input_ids.size()[0], 1))
                esb_logits = torch.cat(probs_boost, dim=1)

            return transformers.modeling_outputs.SequenceClassifierOutput(
                loss=esb_loss,
                logits=esb_logits,
                hidden_states=esb_hidden_states,
                attentions=esb_attentions,
            )
        else:
            # 训练时，返回多个子模型的预测结果
            if args.perturb_attention:
                # aux_outs: 是包含num_models个tensor的list，每个tensor [b_size, num_heads, seq_len, seq_len]
                return aux_outs, transformers.modeling_outputs.SequenceClassifierOutput(loss=all_loss, logits=all_logits, hidden_states=all_hidden_states, attentions=all_attentions)
            else:
                return transformers.modeling_outputs.SequenceClassifierOutput(loss=all_loss, logits=all_logits, hidden_states=all_hidden_states, attentions=all_attentions)