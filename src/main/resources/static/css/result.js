window.onload = function () {
//    var ctx = document.getElementById('dougnutChart').getContext('2d');
//    window.myDoughnut = new Chart(ctx, config);

//    var lctx = document.getElementById('barChart').getContext('2d');
//    window.myLine = new Chart(lctx, lconfig);
    getData("");
    getSimilarData();
};

function addNew() {
    window.location = "./add.html";
}

function goToHome() {
    window.location = "./home.html";
}



var items = [];

$(function () {
    $("#grid").jsGrid({
        width: "100%",
        height: "150px",
        filtering: false,
        editing: false,
        sorting: true,
        paging: true,
        data: items,
        fields: [{
                name: "ID",
                type: "text",
                width: 30
            },
            {
                name: "Recommendation",
                type: "text",
                width: 50
            }
        ]
    });
});

$(function () {
    $("#similar_grid").jsGrid({
        width: "50%",
        height: "300px",
        filtering: false,
        editing: false,
        sorting: true,
        paging: true,
        data: items,
        fields: [{
                name: "ID",
                type: "text",
                width: 30
            }
        ]
    });
});

function getData(url) {
	var va = document.getElementById("retain");
	var recommentation = JSON.parse(va.textContent).Result.recommendation
	var newData = {
            ID: "14",
            Recommendation: recommentation,
        };
        $("#grid").jsGrid("insertItem", newData).done(function () {
            console.log("insertion completed");
        });
}

function getSimilarData() {
	var va = document.getElementById("retain");
	var similar_emp = JSON.parse(va.textContent).Result.similaremployees;
	var str_array = similar_emp.split(',');
	var similar_list = [];
	var count = 0;
	for(var i = 0; i < str_array.length; i++) {
		if(count <6){
			if(similar_list.indexOf(str_array[i]) == -1){
				count ++;
				 $("#similar_grid").jsGrid("insertItem",{ID:str_array[i]}).done(function () {
				        console.log("insertion completed");
				    });
			}
			similar_list.push(str_array[i])
		}else{
			break;
		}
	}
   
}