from skimage.io import imread
from skimage.transform import resize
class SofaDataset(torch.utils.data.Dataset):
    # python run_reconstruction.py --name sofa_hd --dataset sofa --batch_size 2
    def __init__(self):
        self.dataset = "sofa"
        self.res = args.image_resolution

    def __len__(self):
        return 3

    def __getitem__(self,index):
        base = "/home/ubuntu/work/convmesh/sofa_data/"
        img_file = base+str(index+1)+'.png'
        img = imread(img_file)
        print("Loading SOFA IMG: ",img.shape, " ",img_file)
        img = resize(img, (self.res,self.res),anti_aliasing=True)
        print("Resized SOFA IMG: ",img.shape)
        
        mask_file = base+str(index+1)+'m.png'
        mask = imread(mask_file)
        print("Loading SOFA MASK: ",mask.shape, " ", mask_file)
        mask = resize(mask, (self.res,self.res),anti_aliasing=True)
        print("Resized SOFA MASK: ",mask.shape)

        # mask = mask[:,:,1]
        # img *= mask[np.newaxis, :, :]
        img *= mask
        img = torch.FloatTensor(img)
        img = img.permute(2,0,1)
        mask = torch.FloatTensor(mask)
        mask = mask.permute(2,0,1)
        ind = torch.LongTensor([index])
        
        extra_imgs = []
        scale = torch.FloatTensor([100.1])
        # trans = torch.FloatTensor([[200.2],[151.1]])
        trans = torch.FloatTensor([0,0.1,0])
        # rot = torch.FloatTensor([[-0.05,-0.05,-0.05],[0.05,0.05,0.05],[0.1,-0.1,0.1]])
        # rot = np.pad(rot, (0,1), 'constant')
        # rot[3,3] = 1
        # rot = torch.FloatTensor([-0.05,0.1,-0.05,0.2])
        rot = torch.FloatTensor([ 0.08282554,  0.65323734,  0.72993609, -0.18334177])
        output = torch.cat((img, mask[:1,:,:]), dim=0)
        # output = output.permute(1,2,0)
        return (output, *extra_imgs, scale, trans, rot, ind)
