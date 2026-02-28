import json

filepath = '/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/CustomDNN/gpt2_tests/training_logs/graph.ipynb'

with open(filepath, 'r') as f:
    notebook = json.load(f)

new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "id": "overlap_graphs",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "df9 = pd.read_csv('/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/CustomDNN/gpt2_tests/training_logs/customdnn_training_log9.csv')\n",
    "df2 = pd.read_csv('/home/blu-bridge005/Desktop/Anuj@BluBridge/TensorParallel/CustomDNN/gpt2_tests/training_logs/customdnn_training_log2.csv')\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(df9['step'][:500].to_numpy(), df9['loss'][:500].to_numpy(), label='Log 9', color='blue', alpha=0.7)\n",
    "plt.plot(df2['step'][:500].to_numpy(), df2['loss'][:500].to_numpy(), label='Log 2', color='red', alpha=0.7)\n",
    "\n",
    "plt.xlabel('Step')\n",
    "plt.ylabel('Loss')\n",
    "plt.title('Training Loss Comparison')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.savefig('combined_loss_curve.png')\n",
    "plt.show()"
   ]
}

notebook['cells'].append(new_cell)

with open(filepath, 'w') as f:
    json.dump(notebook, f, indent=1)

print("done")
