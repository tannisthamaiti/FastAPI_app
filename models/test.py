import os
from os.path import join as pjoin
from .data_loader import *
from .metrics import runningScore
from . import SeismicNet4EncoderASPP
import torch.nn.functional as F
import torchvision.utils as vutils

def patch_label_2d(model, img, patch_size, stride):
    img = torch.squeeze(img)
    h, w = img.shape  # height and width

    # Pad image with patch_size/2:
    ps = int(np.floor(patch_size / 2))  # pad size
    img_p = F.pad(img, pad=(ps, ps, ps, ps), mode='constant', value=0)

    num_classes = 6
    output_p = torch.zeros([1, num_classes, h + 2 * ps, w + 2 * ps])

    # generate output:
    for hdx in range(0, h - patch_size + ps, stride):
        for wdx in range(0, w - patch_size + ps, stride):
            patch = img_p[hdx + ps: hdx + ps + patch_size,
                    wdx + ps: wdx + ps + patch_size]
            patch = patch.unsqueeze(dim=0)  # channel dim
            patch = patch.unsqueeze(dim=0)  # batch dim
            # patch_img = to_3_channels(patch)

            assert (patch.shape == (1, 1, patch_size, patch_size))
            # edited by Tannistha
            # assert (patch_img.shape == (1, 3, patch_size, patch_size))

            model_output = model(patch)
            # model_output = model(patch_img)
            output_p[:, :, hdx + ps: hdx + ps + patch_size, wdx + ps: wdx +
                                                                      ps + patch_size] += torch.squeeze(
                model_output.detach().cpu())

    # crop the output_p in the middke
    output = output_p[:, :, ps:-ps, ps:-ps]
    return output

def predict_section(idx):
    test_set = SectionLoader(is_transform=True, split='test1',augmentations=None)

    test_loader = data.DataLoader(test_set,batch_size=1,num_workers=1,shuffle=False)

    n_classes = test_set.n_classes

    model = getattr(SeismicNet4EncoderASPP, 'seismicnet')()

    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(pjoin("./best_checkpoint.pth"), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    running_metrics_overall = runningScore(n_classes)
    image_original = test_set[idx][0]
    labels_original = test_set[idx][1]

    with torch.no_grad():  # operations inside don't track history

        model.eval()
        outputs = patch_label_2d(model=model, img=image_original, patch_size=99, stride=50)
        pred = outputs.detach().max(1)[1].numpy()
        gt = labels_original.numpy()
        running_metrics_overall.update(gt, pred)
    PA = running_metrics_overall.get_scores()[0]["Pixel Acc: "]
    CA = running_metrics_overall.get_scores()[0]["Class Accuracy: "]
    MCA = running_metrics_overall.get_scores()[0]["Mean Class Acc: "]
    FWIoU = running_metrics_overall.get_scores()[0]["Freq Weighted IoU: "]
    MIoU = running_metrics_overall.get_scores()[0]["Mean IoU: "]
    IoU = running_metrics_overall.get_scores()[1]

    # tb_original_image = vutils.make_grid(image_original[0], normalize=True, scale_each=True)
    # original_image = tb_original_image.permute(1, 2, 0).numpy()
    # fig, ax = plt.subplots(figsize=(14, 8))
    # ax.imshow(original_image)
    # labels_original = labels_original.numpy()[0]
    # correct_label_decoded = test_set.decode_segmap(np.squeeze(labels_original))
    # fig, ax1 = plt.subplots(figsize=(14, 8), )
    # ax1.imshow(correct_label_decoded)
    # out = F.softmax(outputs, dim=1)
    # prediction = out.max(1)[1].cpu().numpy()[0]
    # decoded = test_set.decode_segmap(np.squeeze(prediction))
    # fig, ax2 = plt.subplots(figsize=(14, 8))
    # ax2.imshow(decoded)
    CA = [A if not np.isnan(A) else 0 for A in CA]
    IoU = [IU if not np.isnan(IU) else 0 for IU in IoU]
    return PA, CA, MCA, FWIoU, MIoU, IoU