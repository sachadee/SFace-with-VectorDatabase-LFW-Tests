
  # 🚀 Optimized Face Recognition: 

A Python complete API implementation leveraging vector database technology for high-speed searches, pre-loaded with the LFW (Labeled Faces in the Wild) dataset for out-of-the-box accuracy. Using Sface https://github.com/zhongyy/SFace for recognition and Yunet https://github.com/Mr-PU/YUNet.

This API use a self developed vector Database to store and compare embedding vectors. 

I provide a fulfilled database with all the LFW dataset (12853 images) embeddings in it for testes purpose.

there is 2 python classes : 

 1. class **FaceRecognitionSystem**
 2.  class **VectorDatabase**
 
## Dependencies:

    cv2
	numpy
	typing
	string
	time
	pybase64
	io
	json
	random
	os

The recognizer as it is need the vectorDatabase class to add, delete, recognize true it

the vectorDatabase class can be used in other project to init it for another project:

    from sdeeVectorDB import VectorDatabase

	db = VectorDatabase(data_dir='vector_db',collection='default', dim=128 )

where `data_dir` is the directory where the databases will be stored and `collection` is the name of the database and `dim` is the size of the embedding vector.





