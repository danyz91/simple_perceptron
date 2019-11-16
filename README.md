# Simple Perceptron
Simple perceptron python implementation for binary operations

## Requirements
- python3
- numpy
- tqdm


## Tutorial
Run main.py in order to evaluate perceptron training on given dataset

### Input parameter
-  `-d --dataset ` REQUIRED 
    Dataset mnemonic on which performing training. Use one in the list [AND,OR,NAND,NOR]  
-  `-l --training_length ` OPTIONAL. 
    Number of epochs of learning stage. Default value is 100
-  `-r --learning_rate ` OPTIONAL 
    Learning rate of perceptrion. Default value is 0.1
    
### Example
 ```
 python3 main.py -d AND
 ```
 

		
