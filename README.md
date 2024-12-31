# LNN_UAV

## 文件夹说明

```bash
demo.mp4    演示视频
-gazebo_model    gazebo仿真模型
-launch    roslaunch文件和世界文件
-scripts    程序文件夹
    communication.py    飞控通信class
    data_collect.py    数据集制备主程序
    detect.py    图像识别class
    proxy.py    运动控制接口库
    Pub_yaw.py    无线电测向模拟发布程序
    standard_vtol_position.py    地面计算飞机位置发布程序
    target_calc.py    目标位置解算库
    inferencer.py    模型正向推理class

    lstm_test.py    lstm测试程序
    lstm_stress_test.py    lstm压力测试程序
    lstminferencer.py    lstm正向推理class

    model_test.py    模型测试程序
    model_test2.py
    stress_test2.py    压力测试程序
    
    .*\.pth    模型权重文件
    .*\.ckpt    训练过程断点文件

    -utili    杂项文件

    -data_dr   
        -pic    图片文件夹

        anal.py    数据分析杂项程序
        anal2.py

        resume_training.py    恢复训练程序
        thermo.py    相关性评估程序
        
        train.py    不同版本的训练程序
        train2.py
        train3.py

        validate.py    不同版本不同用途的数值测试程序
        validate2.py
        validate3.py
        validate_ltm.py

        .*\.npy    数据集文件
        .*\.pth    模型权重文件
        .*\.ckpt   训练过程断点文件
```

## 部署

#### 1、ros环境配置
```
wget http://fishros.com/install -O fishros && . fishros
```

#### 2、px4环境配置

- 详见[xtdrone文档](https://www.yuque.com/xtdrone/manual_cn)

#### 3、编译本ros包
- 将文件夹中gazebo模型文件夹中的模型拷贝至Gazebo模型路径
- 将文件夹拷贝至ros工作空间
```bash
catkin build
```

#### 4、启动！
```bash 
source .bashrc
roslaunch lnn_landing LNN_test.launch
```
正常情况下会出现类似水磨石花纹的地板和一架白色飞机，若飞机为黑色则检查是否成功替换模型文件

另开一个窗口
```bash
python3 model_test2.py
```
稍作等待后会出现飞机下视摄像头视角的cv2窗口，若没有出现则检查gazebo_ros插件是否配置成功。飞机会在稍后解锁起飞，若没有起飞请下载QGC地面站将飞机的arm检查条件调至较宽松情况并设置无遥控器信号允许arm。

#### 演示视频
<video width="640" height="360" controls>
  <source src="/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


##### 有问题请联系 91mrqiao@mail.nwpu.edu.cn