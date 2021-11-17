import torch

class Permutation_matrix(torch.nn.Module):
    def __init__(self, imsize: int, noise: float=0, perm_type: str='rotation'):
        super().__init__()
        if perm_type=='rotation':
            num_pixels = imsize * imsize
            self.perm_matrix = torch.zeros(num_pixels, num_pixels, dtype=torch.float)
            for i in range(num_pixels): # ith element --> (x,y)th element in image when i = y * imsize + x
                y = int(i / imsize)
                x = i % imsize
                new_x = (-(1+y)) % imsize
                new_y = x
                new_i = new_y * imsize + new_x
                self.perm_matrix[new_i][i] = 1.0
        self.horizontal = self.perm_matrix.clone()

        if noise:
            self.perm_matrix += noise * torch.rand(self.perm_matrix.shape)

        self.perm_matrix = torch.nn.Parameter(self.perm_matrix)

    def forward(self, img):
        img_reshape = img.reshape(img.shape[0], img.shape[1],-1)
        img_reshape = img_reshape.unsqueeze(3)
        out = torch.matmul(self.perm_matrix, img_reshape) # batch matrix multiplication
        out.squeeze(3)
        out = out.reshape(*img.shape)

        return out





