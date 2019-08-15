# pytorch-deeplab-xception
- testBatch.py 测试一个batch,带标注label,给出测试指标  
- testPic.py 读入测试list写出分割结果  
#### rgb-d + 320x200  
- /data/sweeper/ bash cp.sh  
  新相机分辨率减少了一倍，效果变差，提出用上depth信息，给出图片中De名字结尾的图，但是跟标注图名字无法完全对应，用此脚本提取，目前是手工删除多余的，以后可以计算两图差值再拷贝
