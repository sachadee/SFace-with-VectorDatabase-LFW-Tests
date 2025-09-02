
  # 🚀 Optimized Face Recognition: 

A Python complete API implementation leveraging vector database technology for high-speed searches, pre-loaded with the LFW (Labeled Faces in the Wild) dataset for out-of-the-box accuracy. Using Sface https://github.com/zhongyy/SFace for recognition and Yunet https://github.com/Mr-PU/YUNet.

This API use a self developed vector Database to store and compare embedding vectors. 

I provide a fulfilled database with all the LFW dataset (12853 images) embeddings generated with the SFace recognition model  for tests purpose.
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

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
 	orjson #Important for VectorDatabase
	json
	random
	os

The recognizer as it is need the vectorDatabase class to add, delete, recognize true it

the vectorDatabase class can be used in other project to init it for another project:

    from sdeeVectorDB import VectorDatabase

	db = VectorDatabase(data_dir='vector_db',collection='default', dim=128 )

where `data_dir` is the directory where the databases will be stored and `collection` is the name of the database and `dim` is the size of the embedding vector.

In case you want use it in one of your project here is the functions and some example.

## vectorDatabase API:

|function  | param | param |param|
|--|--|--|--|
|`add_vector`  | `vector` float32 numpy array |`metadata` json dict | |
|`batch_add_vectors` | `vectors` array of float32 numpy arrays | `metadatas` array of json dicts| |
|`search_vectors`|`query vector` float32 numpy array|`k` int of the number of results desired within the threshold|`threshold` float 0.46 default (cosine)|
|`get`|`where` Optional json dict key:value to  search in the database | `incl`Optional ['vector','metadata'] the desired element(s) to include in the response||
|`getTotalFaces`|||
|`get_vector`|`vector_id` int index of the vector to get||
|`delete_vector`|`vector_id`|int index of the vector to delete|
|`deleteAllFiltered`|`where` json dict key:value||

**Examples**:

	from sdeeVectorDB import VectorDatabase

	db = VectorDatabase(data_dir='vector_db',collection='example', dim=6 )
	
	#add_vector:
	
    embedding = np.array([ 0.6162391  -1.5986143  -2.0916946   0.12942934  1.1227669  -1.4042859],dtype=np.float32)
	metadata = {"uuid":"AsRtfgW3e4","name": "John Doe","location": "London"}

	db.add_vector(embedding, metadata)
	
	#batch_add_vectors:

    embedding1 = np.array([ 0.6162391  -1.5986143  -2.0916946   0.12942934  1.1227669  -1.4042859],dtype=np.float32)
    embedding2 = np.array([1.8788961   0.671485   -1.7423328   0.7557847   0.51548755  1.8309314],dtype=np.float32)

	metadata1 = {"uuid":"AsRtfgW3e4","name": "John Doe","location": "London"}
	metadata2 = {"uuid":"BgHareD1z6","name": "Jane Doe","location": "Paris"}
	
	db.batch_add_vectors([embedding1,embedding2],[metadata1,metadata2])

	#search_vectors:

	embedding = np.array([ 0.6162391  -1.5986143  -2.0916946   0.12942934  1.1227669  -1.4042859],dtype=np.float32)

	result = db.search_vectors(embedding,k=1,threshold=0.5)
	print(result)
	#Match
	#[{'id': 51, 'similarity': 0.59669983, 'metadata': {'id': 'a6dd92be-8547-45e4-9585-afb2eb91aaf6', 'name': 'kevin doe'}}]
	#No Match
	{'status': 'success', 'msg': 'unrecognized'}  

	#get :

    res = db.get(where={"name": "Aaron_Guiel"},incl=["metadata"])
	print(res)
	#Found
	#[{'id': 1, 'metadata': {'id': 'bbf6f814-b7e8-4530-a986-54f295a4cdb5', 'name': 'Aaron_Guiel'}}]
	#Not Found
	#[]

	res = db.get(where={"name": "Aaron_Guiel"},incl=["vector"])

	print(res)

	#Found
	#[{'id': 1, 'vector': [-1.9369930028915405, -2.049903154373169, 0.22178973257541656, -0.5580120086669922, 0.06811564415693283, 0.5701966285705566]}]

	res = db.get(incl=["metadata"])
	#res contain all metadatas
	
	res = db.get(incl=["vector"])
	#res contain all vectors

 	#deleteAllFiltered delete all faces matching name
  	res = db.deleteAllFiltered(where={"name": John Doe'})
   
when using the face `recognizer` all these function are predefined in the class:


## The Face Recognizer API

Basically:

	face_system.add_face(image,name)
	face_system.del_face(name)
	res = face_system.match(image,name,threshold=0.46)
	res = face_system.recognize(image,k=5,threshold=0.46) #k=5 to get the five best results inside the threshold.
 
 	##the add_face function will automatically make these 2 functions
    aligned_face = face_system.detect_face(image)
 	embedding = face_system.extract_embedding(aligned_face)

 to init your own vectorDatabase just edit the file recognizer.py : and chanche the `collection` name in the class init part :

    def __init__(self, 
              detector_path='./model/face_detection_yunet_2023mar.onnx',
              recognizer_path='./model/face_recognition_sface_2021dec_int8bq.onnx',
              db_path='vector_db',
              collection='Your_Collection_Name', #### Put the new collection Name here
        	  cropped_dir='croppedFaces',
              target_size=320):

The recognizer file is well documented!


You can access the VectorDatabse functions true `.vecDb`

    tot_face = face_system.vecDb.getTotalFaces()
	print(tot_face)
 	###12848

  	aligned_face = face_system.detect_face(image)
	embedding = face_system.extract_embedding(aligned_face)
 	metadata = {"uuid":"AsRtfgW3e4","name": "John Doe","location": "London"}

	face_system.vecDb.add_vector(embedding,metadata) 
 	 ##Similar as :
 	face_system.add_face(image,name)
 
just download the codes and run :

> >> **python demo.py**

