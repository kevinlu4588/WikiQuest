import seaborn as sns
import matplotlib.pyplot as plt

# Sample data (replace this with your own dataset)
data = sns.load_dataset("tips")

# Create a violin plot
sns.violinplot(x="day", y="total_bill", data=data)

# Show the plot
plt.show()