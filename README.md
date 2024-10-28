# zero shot yolo

you only look o̶n̶c̶e̶ zero times

zero shot detector that generalises to new objects by including reference images
of the object of interest as part of the model input.

see https://matpalm.com/blog/yolz for a walkthrough

![highlevel diagram of model](model.png)

## high level repro

```
sh generate_split_chunk_data.sh
sh v3_train.sh
sh v3_test.sh
```