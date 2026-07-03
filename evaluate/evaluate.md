# Evaluation of QuNoC

## Experiment Setup

**Simulation method**

默认设置下，测试100\*100的NoC，最大计算芯粒尺寸为10\*10，最长传输芯粒为80，片间耦合器稀疏度为3，片间耦合器保真度为0.983，延迟为200\,ns，片内耦合器保真度为0.990，延迟为25\,ns。测量操作的保真度为0.990，延迟为2280\,ns。EQST的保真度和延迟分别根据距离进行建模，保真度呈指数衰减，延迟为25d\,ns，其中d为源物理量子比特和目标物理量子比特之间的拓扑距离。

**Baseline**

MECH，保证相同NoC尺寸下，用于计算的比特区域数量相同。

**Benchmarks**

QFT、VQE、QAOA和BV算法，分别代表不同类型的量子程序，包含不同的量子门结构和连接模式。对于不同规模的NoC，按计算区域的比例测试不同规模的量子程序，在100\*100的NoC包，默认配置为220个量子比特的程序。

**Metric**

- 平均门保真度：程序编译后，每个原始两量子比特门的平均保真度，反映了长距离通信引入的退化程度，提供了程序可靠性的实际指标。
- 电路执行时间：整个程序的执行时间，反映了在理想调度下的延迟。

## Overall Result

先用template/greedy/heuristic/rl策略设计40\*40到120\*120的NoC包。

template：使用预定义strip/radial/grid三种的NoC包设计模板，选取最优的一个。

再测试在该架构上QuNoC编译的效果，对比MECH，在40\*40到120\*120的5种不同规模的NoC上测试4种不同类型的量子程序。

## Scalability Analysis

先用heuristic策略设计不同规模的NoC包，再测试编译结果的变化趋势。

**Quantum NoC package shape**

测试不同长宽比的NoC对编译结果的影响，保持总量子比特数10000不变，长宽比从1:1到1:100。

**Qubit capacity of quantum chiplet**

测试不同最大量子芯粒尺寸对编译结果的影响，最大量子芯粒的尺寸从5\*5到25\*25，传输芯粒的长度按比例缩放。

**Qubit topology in compute die**

测试不同量子芯粒内量子比特拓扑结构对编译结果的影响，量子芯粒内部拓扑结构类型：Square、Hexagon、Heavy-square和Heavy-hexagon。

**Inter-die coupler sparsity**

测试不同稀疏度的片间耦合器对编译结果的影响，片间耦合器稀疏度从1到5。

**Inter-die coupler fidelity**

测试不同片间耦合器保真度对编译结果的影响，片间耦合器保真度从0.96到0.99。

## Ablation Study

**Necessity of Sensate for quantum NoC**

在40\*40至120\*120的NoC上，对比 MECH with EQST 和 Sensate without optimization，来验证EQST和协调编译机制对于性能提升的贡献。

MECH with EQST：将highway上构建GHZ态，替换为沿highway依次传输量子比特来构建GHZ态，EQST只在片内进行，需要片间通信时，仍然使用传统的片间通信方式，即多跳通信。
Sensate without optimization：不使用编译优化，所有传输操作贪心发射。

**Decomposition of compiled hardware operations**

对Overall Result中40\*40至120\*120的NoC上编译结果进行分解，统计不同类型的硬件操作数量，来验证EQST在替代传统片间通信方式方面的作用。

**Initial qubit placement strategy**

对比不同初始量子比特放置策略的编译结果，来验证初始量子比特放置对于编译性能的影响。
