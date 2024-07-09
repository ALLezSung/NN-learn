import matplotlib.pyplot as plt
import numpy as np
import time

# 开启交互模式
plt.ion()

# 创建一个图像窗口
fig, ax = plt.subplots()
line, = ax.plot(np.random.randn(100))

# 更新图像的循环
for _ in range(10):
    # 更新数据
    line.set_ydata(np.random.randn(100))
    # 重新绘制图像
    fig.canvas.draw()
    fig.canvas.flush_events()
    # 暂停一会儿，模拟长时间运行的计算
    time.sleep(1)

# 关闭交互模式
plt.ioff()
# 显示最终图像
plt.show()