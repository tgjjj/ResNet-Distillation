import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from tensorboardX import SummaryWriter
import ResNet_plain
import ResNet

def testCIFAR100(net, testLoader):
	net.eval()
	correct = 0
	total = 0
	totalTime = 0.0
	with torch.no_grad():
		for sampleBatch in testLoader:
			images = sampleBatch[0]
			images = images.to(device)
			labels = sampleBatch[1]
			labels = labels.to(device)
			torch.cuda.synchronize()
			start = time.time()
			out = net(images)
			torch.cuda.synchronize()
			end = time.time()
			# print(end - start)
			totalTime += (end - start)
			_, pred = torch.max(out[-1], 1)
			correct += (pred == labels).sum().item()
			total += labels.size(0)
	acc = float(correct) / total
	# print("Accuracy = %.4f" % acc)
	print("Test on CIFAR100 cost %.6f seconds" % totalTime)
	net.train()
	return acc

def trainCIFAR100(net, trainLoader, testLoader):
	writer = SummaryWriter()
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)
	lossCount = 0
	for epoch in range(0, 160):
		print("--------------------------------------")
		print("Epoch: " + str(epoch))
		if epoch == 120 or epoch == 145:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * 0.1
		print("Learning Rate = %.6f" % optimizer.param_groups[0]['lr'])
		losses = 0.0
		for i, data in enumerate(trainLoader, 0):
			imgs, labels = data
			imgs = imgs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = net(imgs)
			loss = criterion(outputs[-1], labels)
			# print(loss)
			loss.backward()
			optimizer.step()

			losses += loss.item()
			if i % showLossPerIter == (showLossPerIter - 1):
				print("Epoch: " + str(epoch) + ", Iter: " + str(i + 1) + ", Loss = %.3f" % (losses / showLossPerIter))
				writer.add_scalar("scalar/Teacher_Loss", losses / showLossPerIter, lossCount)
				losses = 0.0
				lossCount += 1
		testAcc = testCIFAR100(net, testLoader)
		trainAcc = testCIFAR100(net, trainLoader)
		print("Train set Acc = %.4f" % trainAcc)
		print("Test  set Acc = %.4f" % testAcc)
		writer.add_scalars("scalar/Teacher_Acc", {"Train" : trainAcc, "Test" : testAcc}, epoch)
	writer.close()

def perPixelLoss(tensor1, tensor2):
	return torch.mean(torch.abs(tensor1 - tensor2))

def distillation(teacher, student, trainLoader, testLoader):
	trainParams = [student.layer1.parameters(), student.layer2.parameters(), student.layer3.parameters()]
	writer = SummaryWriter()
	criterion = torch.nn.CrossEntropyLoss()
	stageNum = 5
	for stage in range(1, stageNum):
		optimizer = torch.optim.SGD(trainParams[stage-1], lr = 0.1, momentum = 0.9) if not stage == stageNum - 1 \
		else torch.optim.SGD(student.parameters(), lr = 0.001, momentum = 0.9)
		lossCount = 0
		for epoch in range(0, 30):
			print("--------------------------------------")
			print("Stage: " + str(stage) + ", Epoch: " + str(epoch))
			if epoch + 1 == 15 or epoch + 1 == 25:
				for param_group in optimizer.param_groups:
					param_group['lr'] = param_group['lr'] * 0.1
			print("Learning Rate = %.6f" % optimizer.param_groups[0]["lr"])
			losses = 0.0
			for i, data in enumerate(trainLoader, 0):
				imgs, labels = data
				imgs = imgs.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()
				outputT = teacher(imgs)
				outputS = student(imgs)
				Loss = perPixelLoss(outputT[stage], outputS[stage]) if not (stage == (stageNum - 1)) else criterion(outputS[stage], labels)
				Loss.backward(retain_graph=False)
				optimizer.step()

				losses += Loss.item()
				if i % showLossPerIter == (showLossPerIter - 1):
					print("Stage: " + str(stage) + ", Epoch: " + str(epoch) + ", Iter: " + str(i + 1))
					print("Loss = %.3f" % (losses / showLossPerIter))
					writer.add_scalar("scalar/Student_Loss_Stage_" + str(stage), losses / showLossPerIter, lossCount)
					losses = 0.0
					lossCount += 1
			teacherAcc = testCIFAR100(teacher, testLoader)
			studentAcc = testCIFAR100(student, testLoader)
			print("Test set Teacher Acc = %.4f" % teacherAcc)
			print("Test set Student Acc = %.4f" % studentAcc)
			writer.add_scalars("scalar/Test_Acc_Stage_" + str(stage), {"Teacher" : teacherAcc, "Student" : studentAcc}, epoch)
	writer.close()


if __name__ == "__main__":
	trainTeacherFlag = False
	distillationFlag = False
	finalTestFlag = True

	rootDir = "./"
	showLossPerIter = 10
	networkPrefix = "ResNet-20"
	modelSavePath = "./models/" + networkPrefix
	modelNameTeacher = networkPrefix + "-teacher.pth"
	modelNameStudent = networkPrefix + "-student.pth"
	teacher = ResNet.resnet20(num_classes = 100)
	student = ResNet_plain.resnet20(num_classes = 100)

	transformTrain = transforms.Compose([transforms.RandomCrop(28), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), \
		transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	transformTest = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	trainSet = torchvision.datasets.CIFAR100(root = rootDir, train = True, download = True, transform = transformTrain)
	trainLoader = torch.utils.data.DataLoader(trainSet, batch_size = 1024, shuffle = True, num_workers = 16)
	testSet = torchvision.datasets.CIFAR100(root = rootDir, train = False, download = True, transform = transformTest)
	testLoader = torch.utils.data.DataLoader(testSet, batch_size = 1024, shuffle = True, num_workers = 16)

	if not os.path.exists(modelSavePath):
		os.mkdir(modelSavePath)

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")

	if trainTeacherFlag:
		teacher = teacher.to(device)
		trainCIFAR100(teacher, trainLoader, testLoader)
		torch.save(teacher, os.path.join(modelSavePath, modelNameTeacher))

	if distillationFlag:
		if not trainTeacherFlag:
			teacher = torch.load(os.path.join(modelSavePath, modelNameTeacher))
		student_dict = teacher.state_dict()
		student.load_state_dict(student_dict)
		student = student.to(device)
		teacherAcc = testCIFAR100(teacher, testLoader)
		studentAcc = testCIFAR100(student, testLoader)
		print("Test set Teacher Acc = %.4f" % teacherAcc)
		print("Test set Student Acc = %.4f" % studentAcc)
		distillation(teacher, student, trainLoader, testLoader)
		torch.save(student, os.path.join(modelSavePath, modelNameStudent))

	if finalTestFlag:
		if not distillationFlag:
			student = torch.load(os.path.join(modelSavePath, modelNameStudent))
			if not trainTeacherFlag:
				teacher = torch.load(os.path.join(modelSavePath, modelNameTeacher))
		for testRound in range(0, 10):
			print("Test Round: " + str(testRound))
			print(networkPrefix + ": ")
			teacherAcc = testCIFAR100(teacher, testLoader)
			print("Accuracy: %.4f" % teacherAcc)
			print(networkPrefix + "-NoRes: ")
			studentAcc = testCIFAR100(student, testLoader)
			print("Accuracy: %.4f" % studentAcc)