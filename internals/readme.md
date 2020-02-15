* Step 1: Install needed Python packages
	- pip install * [Test from fresh system and add entries here]

* Step 2: Start REST application
	- $SRC_DIR\internals>python RetentionApp.py
	
* Step 3: Launch Application
	- http://127.0.0.1:5000/sap/upload
	- http://127.0.0.1:5000/sap/predict
	- http://127.0.0.1:5000/sap/predict/cluster	

* Step 4: Test from postman

	- Sample input JSON for upload
		{
		  "classLabel" : "Type",
		  "data": [
			{
			  "Type":"Novel",	
			  "BookTitle": "Leading",
			  "BookID": "56353",
			  "BookAuthor": "Sir Alex Ferguson"
			},
			{
			  "Type":"Technical",	
			  "BookTitle": "How Google Works",
			  "BookID": "73638",
			  "BookAuthor": "Eric Smith"
			},
			{
			  "Type":"Novel",
			  "BookTitle": "The Merchant of Venice",
			  "BookID": "37364",
			  "BookAuthor": "William Shakespeare"
			},
			{
			  "Type":"Technical",
			  "BookTitle": "Java 8",
			  "BookID": "37564",
			  "BookAuthor": "James Gosling"
			},
			{
			  "Type":"Technical",
			  "BookTitle": "SAP",
			  "BookID": "57364",
			  "BookAuthor": "Me"
			},
			{
			  "Type":"Technical",
			  "BookTitle": "Oracle",
			  "BookID": "37374",
			  "BookAuthor": "Donald"
			},
			{
			  "Type":"Novel",	
			  "BookTitle": "Leading12",
			  "BookID": "59353",
			  "BookAuthor": "Sir Alex Ferguson"
			},
			{
			  "Type":"Technical",	
			  "BookTitle": "How DB Works",
			  "BookID": "83638",
			  "BookAuthor": "Eric Smith"
			},
			{
			  "Type":"Novel",
			  "BookTitle": "The Spy leaves next door",
			  "BookID": "36364",
			  "BookAuthor": "Shyam Benegal"
			},
			{
			  "Type":"Novel",
			  "BookTitle": "Iron Man",
			  "BookID": "76364",
			  "BookAuthor": "XYZ"
			}    
		  ]
		}
	
	Response JSON:
		{
			"success": True,
			"modelname": XYZ,
			"modelaccuracy":XX.YY
		}	
	
	- Sample input JSON for predict
		{
		  "data": [
			{
			  "BookTitle": "Leading",
			  "BookID": "76353",
			  "BookAuthor": "Sir Alex Ferguson"
			},
			{
			  "BookTitle": "How Google Works",
			  "BookID": "83638",
			  "BookAuthor": "Eric Smith"
			},
			{
			  "BookTitle": "The Merchant of Venice",
			  "BookID": "57364",
			  "BookAuthor": "William Shakespeare"
			},
			{
			  "BookTitle": "Java 8",
			  "BookID": "57564",
			  "BookAuthor": "James Gosling"
			},
			{
			  "BookTitle": "SAP",
			  "BookID": "77364",
			  "BookAuthor": "Me"
			},
			{
			  "BookTitle": "Oracle",
			  "BookID": "57374",
			  "BookAuthor": "Donald"
			},
			{
			  "BookTitle": "Leading12",
			  "BookID": "79353",
			  "BookAuthor": "Sir Alex Ferguson"
			},
			{
			  "BookTitle": "How DB Works",
			  "BookID": "93638",
			  "BookAuthor": "Eric Smith"
			},
			{
			 "BookTitle": "The Spy leaves next door",
			  "BookID": "56364",
			  "BookAuthor": "Shyam Benegal"
			},
			{
			  "BookTitle": "Iron Man",
			  "BookID": "96364",
			  "BookAuthor": "XYZ"
			}    
		  ]
		}
	Response Output:
		[
			{
				"prediction": "Technical",
				"BookTitle": "Leading",
				"BookID": 76353,
				"BookAuthor": "Sir Alex Ferguson"
			},
			{
				"prediction": "Technical",
				"BookTitle": "How Google Works",
				"BookID": 83638,
				"BookAuthor": "Eric Smith"
			},
			{
				"prediction": "Technical",
				"BookTitle": "The Merchant of Venice",
				"BookID": 57364,
				"BookAuthor": "William Shakespeare"
			},
			{
				"prediction": "Novel",
				"BookTitle": "Java 8",
				"BookID": 57564,
				"BookAuthor": "James Gosling"
			},
			{
				"prediction": "Technical",
				"BookTitle": "SAP",
				"BookID": 77364,
				"BookAuthor": "Me"
			},
			{
				"prediction": "Technical",
				"BookTitle": "Oracle",
				"BookID": 57374,
				"BookAuthor": "Donald"
			},
			{
				"prediction": "Technical",
				"BookTitle": "Leading12",
				"BookID": 79353,
				"BookAuthor": "Sir Alex Ferguson"
			},
			{
				"prediction": "Technical",
				"BookTitle": "How DB Works",
				"BookID": 93638,
				"BookAuthor": "Eric Smith"
			},
			{
				"prediction": "Novel",
				"BookTitle": "The Spy leaves next door",
				"BookID": 56364,
				"BookAuthor": "Shyam Benegal"
			},
			{
				"prediction": "Novel",
				"BookTitle": "Iron Man",
				"BookID": 96364,
				"BookAuthor": "XYZ"
			}
		]	
		
	- Sample input JSON for clustering prediction results
		{
              "classlabel":"prediction",
              "classvalue":"Yes",
              "id":"BookID",
              "noofclusters": 3
		}
		
	Response Output:			
	[
    {
		 "prediction": "Technical",
		 "BookTitle": "Leading",
		 "BookID": 76353,
		 "BookAuthor": "Sir Alex Ferguson"
        "cluster": 0
    },
	……………………
    {
		 "prediction": "Novel",
		 "BookTitle": "The Spy leaves next door",
		 "BookID": 56364,
		 "BookAuthor": "Shyam Benegal"
        "cluster": 1
    },
	……………………
	{
		 "prediction": "Novel",
		 "BookTitle": "Iron Man",
		 "BookID": 96364,
		 "BookAuthor": "XYZ"
        "cluster": 2
    }
	……………………
	]

* Highlights:
    1. Generic way to upload training dataset (Ex: Till 2018)
	2. For given training dataset; identify best classification algorithms (Ex: Random Forest, Multinomial Naive Bayes, Logistic Regression) using 	  K-Fold classifier techniques and create an attrition classifier model out of that.
    3. Generic way to upload prediction dataset (Ex: 2019)
    4. Various statistical charts based on above prediction using SAP Analytics Cloud
    5. Evaluation of clustering algorithm in grouping similar employees based on above prediction (attributes comparisons using various distance 	  metrics Ex: Manhattan, Euclidian, MahalaNobis)
    6. Evaluating organization community graph in knowing how employee is important at an organization level, based on this manager can decide to 	  retain an employee or not.
