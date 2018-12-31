# linear_functions_prediction

## model summary
<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 100)               200       
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_2 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_3 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 101       
=================================================================
Total params: 30,601
Trainable params: 30,601
Non-trainable params: 0
_________________________________________________________________
</pre>

## training result
Loss and validation loss.   
![train result](train.png?raw=true "train result")

## evaluation result
Prediction of linear function. Seems that the sigmoid function is not suitable for linear functions.   
![evaluation result](eval.png?raw=true "evaluation result")