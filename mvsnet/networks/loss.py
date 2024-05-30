import torch.nn.functional as F

def multi_stage_loss(outputs, labels, masks, weights, Flow1, Flow2):
	tot_loss = 0.
	if Flow1 and Flow2 == False:
		num_stage = 2
		weights = [0.5, 1.0]
	elif Flow1 and Flow2:
		num_stage = 4
	else:
		num_stage = 1
		weights = [1.0]

	for stage_id in range(num_stage):
		depth_i = outputs["stage{}".format(stage_id+1)]["depth"]
		label_i = labels["stage{}".format(stage_id+1)]
		mask_i = masks["stage{}".format(stage_id+1)].bool()
		depth_loss = F.smooth_l1_loss(depth_i[mask_i], label_i[mask_i], reduction='mean')
		tot_loss += depth_loss * weights[stage_id]
	return tot_loss