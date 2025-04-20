import torch  
from torch import nn
from efficientnet_pytorch import EfficientNet
from inplace_abn import InPlaceABN
from EfficientPS_modified import TwoWayFpn, SemanticHead

output_feature_size = {
    0: [16, 24, 40, 112, 320],   # B0
    1: [16, 24, 40, 112, 352],   # B1
    2: [16, 24, 48, 120, 352],   # B2
    3: [24, 32, 48, 136, 384],   # B3
    4: [24, 32, 56, 160, 448],   # B4
    5: [24, 40, 64, 176, 512],   # B5
    6: [32, 40, 72, 200, 576],   # B6
    7: [32, 48, 80, 224, 640],   # B7
    8: [32, 56, 88, 248, 704]    # B8 (if applicable)
}

def generate_backbone_EfficientPS(net_id, use_pretrain=True):
    """
    Create an EfficientNet model base on this repository:
    https://github.com/lukemelas/EfficientNet-PyTorch

    Modify the existing Efficientnet base on the EfficientPS paper,
    ie:
    - replace BN and swish with InplaceBN and LeakyRelu
    - remove se (squeeze and excite) blocks
    Args:
    - cdg (Config) : config object
    Return:
    - backbone (nn.Module) : Modify version of the EfficentNet
    """

    if use_pretrain:
        backbone = EfficientNet.from_pretrained(f'efficientnet-b{net_id}')
    else:
        backbone = EfficientNet.from_name(f'efficientnet-b{net_id}')

    backbone._bn0 = InPlaceABN(num_features=backbone._bn0.num_features, eps=0.001)
    backbone._bn1 = InPlaceABN(num_features=backbone._bn1.num_features, eps=0.001)
    backbone._swish = nn.Identity()
    for i, block in enumerate(backbone._blocks):
        # Remove SE block
        block.has_se = False
        # Additional step to have the correct number of parameter on compute
        block._se_reduce =  nn.Identity()
        block._se_expand = nn.Identity()
        # Replace BN with Inplace BN (default activation is leaky relu)
        if '_bn0' in [name for name, layer in block.named_children()]:
            block._bn0 = InPlaceABN(num_features=block._bn0.num_features, eps=0.001)
        block._bn1 = InPlaceABN(num_features=block._bn1.num_features, eps=0.001)
        block._bn2 = InPlaceABN(num_features=block._bn2.num_features, eps=0.001)

        # Remove swish activation since Inplace BN contains the activation layer
        block._swish = nn.Identity()

    return backbone

class EfficientPS(nn.Module):
    """
    EfficientPS model based on http://panoptic.cs.uni-freiburg.de/
    Implemented as a standard PyTorch nn.Module.
    """
    def __init__(
        self,
        net_id: int,
        classes: int,
        image_size: tuple = (512, 256), 
    ):
        super().__init__()
        self.net_id = net_id
        self.classes = classes
        self.image_size = image_size

        # Initialize backbone, FPN, and semantic head
        self.backbone = generate_backbone_EfficientPS(net_id)
        self.fpn = TwoWayFpn(output_feature_size[net_id])
        self.semantic_head = SemanticHead(self.classes)

        # Debug: Print feature shapes
        self._debug_feature_shapes()

    def _debug_feature_shapes(self):
        """Print the shapes of backbone output features for debugging."""
        self.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.image_size[0], self.image_size[1])
            features = self.backbone.extract_endpoints(dummy_input)
            print("Output feature shapes:")
            for key, value in features.items():
                print(f"{key}: {value.shape}")
        self.train()

    def forward(self, x, targets=None):
        """
        Forward pass for prediction and optional loss computation.
        Args:
            x (torch.Tensor): Input images [batch_size, 3, H, W]
            targets (dict, optional): Ground truth for computing losses (e.g., {'semantic': tensor})
        Returns:
            If targets is None:
                dict: Predictions {'semantic': logits}
            Else:
                tuple: (predictions, losses)
        """
        # Feature extraction
        features = self.backbone.extract_endpoints(x)
        pyramid_features = self.fpn(features)
        outputs, losses = self.semantic_head(pyramid_features, targets)

        if targets is None:
            return outputs
        return outputs, losses