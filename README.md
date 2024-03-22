# 运行命令
`python all.py --device cuda:0 --columns contet --model resultModel --xlsx_directory ./excel --directory ./csv --threshold 0.85`
# 参数说明
device:运行设备，默认使用GPU0(使用CPU就改为"--device cpu")  
columns:输入的excel列名，默认使用contet列  
model:模型路径  
xlsx_directory:输入文件夹路径默认“./excel”  
directory:输出文件夹路径默认“./csv”  
threshold:相似度计算阈值默认0.85  
# 函数解释
## all.py 计算相似度文件
> model = SentenceTransformer(args.model).to(args.device)  
加载模型，并将其移动到指定的设备上

> read()  
读取输入文件夹(excel)中的所有文件，并返回一个包含文件路径和文件内容的列表  
具体实现：  
获取指定目录下的所有 Excel 文件列表。  
遍历每个 Excel 文件，读取其数据，并处理缺失值、特殊字符。格式化列。  
根据 Excel 文件中的一个列（"体系"）进行分组。  
将每个分组的数据保存为单独的 CSV 文件，以 "体系" 列的值作为文件名。

> merge_file()  
> 读取格式化后的excel文件  
> 对同一个分组下条款内容配对  
> 并行计算相似度(pall_func())  
> 并保存到结果文件目录  
