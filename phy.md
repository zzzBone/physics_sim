# 模型



模型大致的发展历程：

| 2016 | **IN**(interaction networks)                       | 交互网络                     |
| ---- | -------------------------------------------------- | ---------------------------- |
| 2018 | **HRN**(Hierarchical Relation Network)             | 层次关系网络                 |
| 2019 | **DPI-Net**(dynamic particle interaction networks) | 动态粒子交互网络             |
| 2020 | **CConv**(continuous convolutions)                 | 连续卷积网络                 |
| 2020 | **GNS**(Graph Network-based Simulators)            | 基于图神经网的模拟器         |
| 2022 | **TIE**(Transformer with Implicit Edges)           | 包含隐式边的Transformer      |
| 2022 | **CGNS**(Constraint-based graph network simulator) | 基于有约束的图神经网的模拟器 |

## 1. TIE

### 1.1 问题定义

将整个系统的状态分成$t$个时间节点，系统中的粒子数为$N$，那么可以将这个系统记为$\chi^{t}=\{x_i^t\}_{i=1}^N$，$x_i^t$是每个在$t$时间下的粒子，粒子的状态可以表示为$x_i^t=[p_i^t,q_i^t,\alpha_i]$，其中的$p_i^t,q_i^t\in R^3$分表表示粒子的位置以及速度，而$\alpha_i\in R^{d_a}$表示粒子属于的类型，比方粒子属于流体或者固体等。

目标是要通过训练数据训练出一个模型$\phi(\cdot)$，这个模型可以从上一个时间节点推测出下一个时间节点的系统状态，即$\hat\chi^{t+1}=\phi(\chi^t)$，那么在拥有初始状态$\chi^0$的情况下就可以得到未来所有时刻的状态。

具体来讲，用于预测的变量只有物体的速度即$\hat{Q}^{t+1}=\{\hat{q}_i^{t+1}\}$，而物体所在时刻的位置则由当前位置和预测得到的速度计算得到，即$\hat{p}_i^{t+1}=p_i^t+\Delta t\cdot \hat{q}_i^{t+1}$。其中的时间间隔$\Delta t$则是由数据集所定的常量。



### 1.2 图神经网络方法

系统中的每个粒子都可以视作图神经网中的一个节点，粒子之间的相互作用可以视作节点之间的有向边。需要注意的是，并不是所有节点之间都需要建立有向边的相互作用，而是可以通过粒子之间的距离选择性忽略一些距离过远的粒子之间的相互作用，具体来说可以使用一个函数进行判断：
$$
g(i,j)=1(||p_i-p_j||_2<R)
$$
其中的$R$是一个提前设置好的关于距离的阈值，$1(\cdot)$是一个指标函数。

图神经网中初始的节点和边由以下方式生成：
$$
v_i^{(0)}=f_V^{enc}(x_i),\\
e_{ij}^{(0)}=f_E^{enc}(x_i,x_j),\\
$$
其中$v_i,e_{ij}\in R^{d_h}$是长度为$d_h$的向量，$f_V^{enc}(\cdot),f_E^{enc}(\cdot)$分别是节点编码器和边编码器。之后，图神经网会进行$L$轮信息传递，计算出每一个时间节点下所有粒子的运动速度：
$$
\begin{aligned}
\boldsymbol{e}_{i j}^{(l+1)} & =f_{E}^{\mathrm{prop}}\left(\boldsymbol{v}_{i}^{(l)}, \boldsymbol{v}_{j}^{(l)}, \boldsymbol{e}_{i j}^{(l)}\right), \\
\boldsymbol{v}_{i}^{(l+1)} & =f_{V}^{\mathrm{prop}}\left(\boldsymbol{v}_{i}^{(l)}, \sum_{j \in \mathcal{N}_{i}} \boldsymbol{e}_{i j}^{(l+1)}\right), \\
\hat{\boldsymbol{q}}_{i} & =f_{V}^{\mathrm{dec}}\left(\boldsymbol{v}_{i}^{(L)}\right),
\end{aligned}
$$
其中$\mathcal{N}_i$表示与第$i$个粒子所邻接的其它所有粒子的集合，$f_E^{prop}(\cdot),f_V^{prop}(\cdot)$分别是节点和边的前向传播网络，$f_V^{dec}(\cdot)$表示节点的解码器，$f_V^{enc}(\cdot),f_E^{enc}(\cdot),f_E^{prop}(\cdot),f_V^{prop}(\cdot),f_V^{dec}(\cdot)$都是由多层感知机（MLPs）构成。



### 1.3 与注意力机制结合的普通方法

具体来讲，图神经网会与注意力机制进行结合，其中进行到第$l$层的计算方式如下：

$$
\begin{aligned}
\omega_{i j} & =\left(W_{Q}^{(l)} \boldsymbol{v}_{i}^{(l)}\right)^{\top} \cdot\left(W_{K}^{(l)} \boldsymbol{v}_{j}^{(l)}\right) \\
\boldsymbol{v}_{i}^{(l+1)} & =\sum_{j} \frac{\exp \left(\omega_{i j} / \sqrt{d}\right)}{\sum_{k} \exp \left(\omega_{i k} / \sqrt{d}\right)} \cdot\left(W_{V}^{(l)} \boldsymbol{v}_{j}^{(l)}\right),
\end{aligned}
$$
其中$W_Q,W_K,W_V$分别是注意力机制中的Query,Key,Value权值矩阵，$d$是特征向量的长度。



### 1.4 改进的方法

如果只是简单将注意力与图神经网络进行结合，实验的结果并不好。作者采用了另一种方式，即将图神经网中的显式边$e_{ij}^{(l+1)}$替换为由两个token组成的隐式边$r_i^{(l)},s_j^{(l)}$，它们分别又称为接收(receiver)token和发送(sender)token：
$$
\begin{aligned}
\boldsymbol{r}_{i}^{(0)} & =W_{r}^{(0)} \boldsymbol{x}_{i}, \quad \boldsymbol{s}_{j}^{(0)}=W_{s}^{(0)} \boldsymbol{x}_{j}, \\
\boldsymbol{r}_{i}^{(l)} & =W_{r}^{(l)} \boldsymbol{v}_{i}^{(l)}+W_{m}^{(l)} \boldsymbol{r}_{i}^{(l-1)}, \\
\boldsymbol{s}_{j}^{(l)} & =W_{s}^{(l)} \boldsymbol{v}_{j}^{(l)}+W_{m}^{(l)} \boldsymbol{s}_{j}^{(l-1)}, \\
\boldsymbol{e}_{i j}^{(l+1)} & =\boldsymbol{r}_{i}^{(l)}+\boldsymbol{s}_{j}^{(l)},
\end{aligned}
$$
因此，每个系统中的粒子都具有三个token，分别是接收(receiver)token，发送(sender)token和状态token$r_i,s_i,v_i$，其中状态token可以使用图神经网络与注意力机制结合的计算方式：
$$
\begin{aligned}
\omega_{i j}^{\prime} & =\left(W_{Q}^{(l)} \boldsymbol{v}_{i}^{(l)}\right)^{\top} \boldsymbol{r}_{i}^{(l)}+\left(W_{Q}^{(l)} \boldsymbol{v}_{i}^{(l)}\right)^{\top} \boldsymbol{s}_{j}^{(l)}, \\
\boldsymbol{v}_{i}^{(l+1)} & =\boldsymbol{r}_{i}^{(l)}+\sum_{j} \frac{\exp \left(\omega_{i j}^{\prime} / \sqrt{d}\right)}{\sum_{k} \exp \left(\omega_{i k}^{\prime} / \sqrt{d}\right)} \cdot \boldsymbol{s}_{j}^{(l)} .
\end{aligned}
$$

再加入归一化操作LayerNorm可以修改上述的公式：

$$
\begin{aligned}
\left(\sigma_{i j}^{(l)}\right)^{2} & =\frac{1}{d}\left(\boldsymbol{r}_{i}^{(l)}\right)^{\top} \boldsymbol{r}_{i}^{(l)}+\frac{1}{d}\left(\boldsymbol{s}_{j}^{(l)}\right)^{\top} \boldsymbol{s}_{j}^{(l)}+\frac{2}{d}\left(\boldsymbol{r}_{i}^{(l)}\right)^{\top} \boldsymbol{s}_{j}^{(l)}-\left(\mu_{r_{i}}^{(l)}+\mu_{s_{j}}^{(l)}\right)^{2}, \\
\omega_{i j}^{\prime \prime} & =\frac{\left(W_{Q}^{(l)} \boldsymbol{v}_{i}^{(l)}\right)^{\top}\left(\boldsymbol{r}_{i}^{(l)}-\mu_{r_{i}}^{(l)}\right)+\left(W_{Q}^{(l)} \boldsymbol{v}_{i}^{(l)}\right)^{\top}\left(\boldsymbol{s}_{j}^{(l)}-\mu_{s_{j}}^{(l)}\right)}{\sigma_{i j}^{(l)}}, \\
\boldsymbol{v}_{i}^{(l+1)} & =\sum_{j} \frac{\exp \left(\omega_{i j}^{\prime \prime} / \sqrt{d}\right)}{\sum_{k} \exp \left(\omega_{i k}^{\prime \prime} / \sqrt{d}\right)} \cdot \frac{\left(\boldsymbol{r}_{i}^{(l)}-\mu_{r_{i}}^{(l)}\right)+\left(\boldsymbol{s}_{j}^{(l)}-\mu_{s_{j}}^{(l)}\right)}{\sigma_{i j}^{(l)}}
\end{aligned}
$$

其中$\mu_{r_i}^{(l)},\mu_{s_i}^{(l)}$分别是接收token和发送token的均值



### 1.5 抽象粒子

为了提高模型的泛用性，模型中还添加了包括材料类型的抽象粒子，比如对于包含$N_a$种材料粒子的系统，模型中需要额外添加$N_a$个抽象粒子记为$A=\{a_k\}_{k=1}^{N_a}$，每个抽象粒子都具有可以训练的状态token，它们的更新方式与普通粒子完全相同，不同的是普通粒子只与其相邻的粒子存在相互作用，而抽象粒子则是强迫其与相同材料属性的所有粒子之间形成相互作用。所有在具有$N_a$种不同材料粒子的模型中需要训练的粒子为$\{a_1,\cdots,a_{N_a},x_1,\cdots,x_N\}$。



### 1.6 训练目标和评价指标

模型使用的训练目标是均方误差（MSE）：
$$
MSE(\hat Q,Q)=\frac{1}{N}\sum_i||\hat{q_i}-q_i||_2^2
$$
评价指标与系统中的粒子材料数量有关，记为$M^3SE$：
$$
M^3SE(\hat Q,Q)=\frac{1}{K}\sum_k\frac{1}{N_k}\sum_i||\hat{q}_{i,k}-q_{i,k}||_2^2
$$
其中$K$表示系统种的粒子材料数量，$N_k$表示第$k$种材料在系统中的所占的粒子总数





## 2. GNS

### 2.1 问题定义

将系统定义为$\chi=\{X^t\}$，$t$为系统中的某个时间节点，物理模型定义为$s:\chi \to \chi$，系统中每个时间节点的状态可以通过递推得到：$\hat{X}^{t_{k+1}}=s(\hat{X}^{t_k})$。

对于可训练的模型$s_\theta$来说，可以通过一个函数进行计算：$d_\theta:\chi\to \mathcal{Y}$，其中的参数$\theta$可以通过训练进行优化，其中的$Y\in\mathcal{Y}$表示系统进行更新过程种的动态语义，比方说在物理模拟中可以表示加速度等信息，在神经网络中可能代表更复杂的语义信息，而且可能存在多种可训练的参数$d_\theta$。那么，物理预测模型可以使用$\hat{X}^{t_{k+1}}=Update(\hat{X}^{t_k},d_\theta)$。

系统中包括$N$个粒子$X=\{x_0,\cdots,x_N\}$，粒子间的相互作用可以理解为相邻粒子间的能量和动量转换。对应在图神经网络中，每个粒子对应一个图中的节点，而粒子之间的相互关系则对应图中的边。

### 2.2 信息传递

模型中的参数$d_\theta$共包含三个步骤：编码器，处理器和解码器。

编码器：$\mathcal{X}\to \mathcal{G}$，将粒子的状态表示为图，即$G^0=Encoder(X)$，其中$G=(V,E,u)$，其中$v_i\in V,v_i=\varepsilon^v(x_i)$是粒子状态嵌入图后的节点表示，$e_{i,j}\in E,e_{i,j}=\varepsilon ^e(r_{i,j})$是粒子之间存在的相互作用嵌入图后的边表示，$u$表示了整个图级别的嵌入，可以表示某些全局属性，比如系统中的重力场，磁场等。

处理器：$\mathcal{G}\to\mathcal{G}$，对编码器生成的初始图进行$M$步更新：$G^M=Processor(G^0)$，其中图的更新由递推得到：$G^{m+1}=GN^{m+1}(G^m)$。

解码器：$\mathcal{G}\to\mathcal{Y}$，将处理器最后生成的图提取出每个节点的动态信息，如加速度等，$y_i=\delta^v(v_i^M)$。



### 2.3 模型的输入与输出

每个粒子输入的状态向量由粒子当前的位置，前$C=5$个时刻的速度，以及粒子属于的材料类型，如水、沙子、粘稠物、刚性、边界粒子，故第$i$个粒子在$t_k$时刻的状态向量为$x_i^{t_k}=[p_i^{t_k},\dot{p}_i^{t_k-C+1},\cdots,\dot{p}_i^{t_k},f_i]$。这里的$C$是一个超参数。如果需要的话，系统的全局属性如外力以及全局材料属性$g$也会作为系统的输入。

模型预测之后的输出为每个粒子的加速度$\ddot p_i$，数据集中使用的数据只有粒子的当前位置$p_i$，速度$\dot p_i$和加速度$\ddot p_i$都可以通过差分进行计算。



### 2.4 编码器

编码器由多层感知机（MLP）构成，记为$\varepsilon ^v,\varepsilon ^e$，分别表示节点和边的编码器，可以将节点和边的特征转换为128维的向量$v_i,e_{i,j}$。

编码器包含两种，其中一种粒子的位置是绝对位置，那么对于节点编码器$\varepsilon^v$的输入即为上述的模型输入，且需要拼接全局属性；而对于边编码器$\varepsilon^e$的输入中则不需要包含任何信息，图神经网络中的初始边是一个可以训练的偏置向量。

另一种则使用粒子的绝对位置，节点编码器$\varepsilon^v$的输入中会屏蔽粒子的位置信息$p_i$，而边编码器$\varepsilon^ e$的输入则需要提供相邻两个粒子的相对位置及大小，即：$r_{i,j}=[(p_i -p_j),||p_i-p_j||]$。

不管使用哪种编码器，全局信息$g$都需要添加到节点编码器的输入$x_i$中。



### 2.5 处理器

处理器中会使用$M$个GN块





## 3. C-GNS

### 3.1 问题定义

## 4.DPI-Net

全称dynamic particle interaction networks，动态粒子交互网络

### 4.1 问题定义





# 数据集

常用的数据集共有四个，分别是BoxBath，FluidFall，FluidShake，RiceGrip。

![sim_BoxBath](E:\课件\硕\科研\图神经网\physics simulation\img\sim_BoxBath.gif)

<center>BoxBath</center>

![sim_FluidFall](E:\课件\硕\科研\图神经网\physics simulation\img\sim_FluidFall.gif)

<center>FluidFall</center>

![sim_FluidShake](E:\课件\硕\科研\图神经网\physics simulation\img\sim_FluidShake.gif)

<center>FluidShake</center>

![sim_RiceGrip](E:\课件\硕\科研\图神经网\physics simulation\img\sim_RiceGrip.gif)

<center>RiceGrip</center>

数据集使用h5文件格式进行保存，每一个h5文件代表了在当前数据集的运动情况下其中一帧中所有粒子（以及场景中受到一定控制的特殊物体）的状态，其中粒子的状态包含位置、速度等信息，下表展示了不同数据集存储的粒子状态的信息类型：

| 数据集类型 | 信息      |            |             |              |              |
| ---------- | --------- | ---------- | ----------- | ------------ | ------------ |
| RiceGrip   | positions | velocities | shape_quats | clusters     | scene_params |
| FluidShake | positions | velocities | shape_quats | scene_params |              |
| BoxBath    | positions | velocities | clusters    |              |              |
| FluidFall  | positions | velocities |             |              |              |

代码中用data_names存储这些信息：

```python
        # data.py 	lines: 845-852
    	if args.env == 'RiceGrip':
            self.data_names = ['positions', 'velocities', 'shape_quats', 'clusters', 'scene_params']
        elif args.env == 'FluidShake':
            self.data_names = ['positions', 'velocities', 'shape_quats', 'scene_params']
        elif args.env == 'BoxBath':
            self.data_names = ['positions', 'velocities', 'clusters']
        elif args.env == 'FluidFall':
            self.data_names = ['positions', 'velocities']       
```

它们代表的含义如下：

- positions：粒子的位置坐标，通常使用空间坐标[x, y, z]表示，但是RiceGrip数据集比较特殊，因为其中包含了弹性材料如糯米，这种特殊的材质除了需要记录粒子当前的位置信息，还应该包括粒子恢复到静止状态后的位置（resting position），故其位置坐标表示为[rest_x, rest_y, rest_z, x, y, z]；

  > *Elastic/Plastic objects.* For elastically deforming particles, only using the current position and velocity as the state is not sufficient, as it is not clear where the particle will be restored after the deformation. Hence, we include the particle state with the resting position to indicate the place where the particle should be restored. When coupled with plastic deformation, the resting position might change during an interaction. Thus, we also infer the motion of the resting position as a part of the state prediction. We use hierarchical modeling for this category but predict next state for each particles individually.

- velocities：粒子的速度，与粒子的位置坐标类似，除RiceGrip数据集之外，其余数据集只需要保存粒子在空间上三个方向的速度即可，记为[xdot, ydot, zdot]，而RiceGrip数据集中的粒子则需要额外添加粒子恢复到静止的速度，记为[rest_xdot, rest_ydot, rest_zdot, xdot, ydot, zdot]；

  positions和velocities合起来称为object_states，即图神经网中节点的状态，是用于训练的最关键的输入数据；

- clusters：对于包含非流体的数据集（BoxBath和RiceGrip），粒子并不是独立运动的，而是需要作为一个整体来看待，相邻的粒子就可以聚成一个类，这个类可以抽象成一个图神经网中的一个节点再进行训练；

  >  For each object that requires modeling of the long-range dependence (e.g. rigid-body), we cluster the particles into several non-overlapping clusters. For each cluster, we add a new particle as the cluster’s root. 

![](https://pdf.cdn.readpaper.com/parsed/fetch_target/417014241673d293c509e2c8937c9598_2_Figure_2_126843620.png)

图片来自于HRN论文



- shape_quats：代表除系统中不受控制的粒子之外，可以进行操控的特定形状（物体）的四元数数据，

- scene_params：用来规定系统中粒子和物体的初始属性，如水滴的位置和半径大小等，并不用于模型训练。

除了上述信息之外，每个数据集还会生成一个stat.h5文件，用作统计所有粒子和特殊物体的位置、速度的均值mean，标准差std和总数count_nodes

读取h5文件可以用python的h5py包进行读取，读取的数据就被称作data

```python
def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data
```



直接从h5文件中读取的上述信息并不是全部用于模型的训练，在这些h5文件被写入dataloader后，模型从中读取的数据包含以下几种：

```python
# train.py	lines: 275 - 278, 309 - 312
for i, data in enumerate(dataloaders[phase]):
	attr, state, rels, n_particles, n_shapes, instance_idx, label = data
	Ra, node_r_idx, node_s_idx, pstep = rels[3], rels[4], rels[5], rels[6]
    ...
    ...
    predicted = model(
                    attr, state, Rr, Rs, Ra, n_particles,
                    node_r_idx, node_s_idx, pstep,
                    instance_idx, phases_dict, args.verbose_model)
```

```python
# data.py	lines: 824 - 952
class PhysicsFleXDataset(Dataset):    
	...
    ...
    def __getitem__(self, idx):
        ...
        ...
		return attr, state, relations, n_particles, n_shapes, instance_idx, label
```

这些数据的说明如下：



- attr：粒子或物体的材质属性，如固体、流体等，是一种类似独热编码的形式，大小为(count_nodes, attr_dim)，每一个节点属于哪一种材质，就将对应的那一列定为1，其余则为0。需要注意的是FluidShake数据集中不同方向的墙壁也属于不同的材质类型。在其中最特殊的是RiceGrip数据集，每个节点还包括三个float类型的属性clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep；

  > The material property parameters clusterStiffness is uniformly sampled between 0.3 and 0.7, clusterPlasticThreshold is uniformly sampled between 1e-5 and 5e-4, and clusterPlasticCreep is uniformly sampled between 0.1 and 0.3. The position of the gripper is randomly sampled within a circle of radius 0.5. 

```python
# data.py	lines: 613 - 615
if args.env == 'RiceGrip':
    # clusterStiffness, clusterPlasticThreshold, cluasterPlasticCreep
    attr[:, -3:] = scene_params[-3:]
```

```python
# data.py	lines: 314 - 320
clusterStiffness = rand_float(0.3, 0.7)
clusterPlasticThreshold = rand_float(0.00001, 0.0005)
clusterPlasticCreep = rand_float(0.1, 0.3)

scene_params = np.array([x, y, z, clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])
pyflex.set_scene(env_idx, scene_params, thread_idx)
scene_params[4] *= 1000.
```



- state：即粒子的positions，velocities，如果是RiceGrip数据集还包含四元数quat属性；

- relations：记录图神经网中边的各种信息，具体如下：


```python
# data.py	lines: 618 - 625
### construct relations
Rr_idxs = []        # relation receiver idx list
Rs_idxs = []        # relation sender idx list
Ras = []            # relation attributes list
values = []         # relation value list (should be 1)
node_r_idxs = []    # list of corresponding receiver node idx
node_s_idxs = []    # list of corresponding sender node idx
psteps = []         # propagation steps
...
...
relations = [Rr_idxs, Rs_idxs, values, Ras, node_r_idxs, node_s_idxs, psteps]
```



Rr_idx和Rs_idx，包括两部分，一部分是粒子与物体之间建立的联系，另一部分是粒子与相邻粒子之间建立的联系。它们的大小为(2, 联系的总数)，第一行是联系的接收者索引（Rr_idx）或者是联系的发送者索引（Rs_idx），第二行则是从0到联系的总数的列表[0, 1, 2, ... , count]





















