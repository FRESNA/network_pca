# network_pca

Principal Component Analysis for PyPSA networks. This little collection of functions intends to create and visualise Principal Components for PyPSA networks.  

For now the code only works with the current 'allocation' branch in pypsa. Please check this out before using. 


Example
```python
import pypsa 
import network_pca as npca


n = pypsa.Network('path/to/pypsa/your/solved_network')
npca.analysis.pcs(n)

npca.plot.components(n, region_data='topology', line_data='flow', plot_colorbar=True, figsize=(16,6), flow_quantile=0.9)

```
![Example of European flow pc](https://user-images.githubusercontent.com/19226431/46303511-4f488680-c5ac-11e8-8115-3e4887cb98ee.png)