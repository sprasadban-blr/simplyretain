function goToHome() {
    window.location = "./home.html";
}

function getRandomColor() {
    var letters = '0123456789ABCDEF';
    var color = '#';
    for (var i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
}

window.onload = function () {
    var names = ["Gender", "Age Category", "Marital Status", "Monthly Income", "Salary Hike", "Perf Rating", "Recognition"];
    var left = document.getElementById('left');
    var w = left.offsetWidth / 3;
    var h = left.offsetHeight / 10;
    for (var i = 0; i < 7; i++) {
        var circle = document.createElement("div");
        circle.className = "circle";
        circle.draggable = true;
        circle.style.backgroundColor = getRandomColor();
        circle.style.width = w;
        circle.style.height = w;
        circle.style.left = Math.floor(Math.random() * (left.offsetWidth - w)) - 20;
        circle.style.top = 50 + Math.floor(Math.random() * (left.offsetHeight - w - 50));
        circle.style.zIndex = i;
        circle.innerText = names[i];
        circle.style.lineHeight = w + 'px';
        left.appendChild(circle);
        circle.addEventListener('drag', dragBubble, false);
        circle.addEventListener('dragend', dropBubble, false);
        circle.addEventListener('click', clickBubble, false);
    }
};

var maxz = 10;

function dragBubble(e) {
    e.preventDefault();
    e.currentTarget.style.border = "2px solid " + e.currentTarget.style.backgroundColor;
    e.currentTarget.style.boxShadow = "0 0 11px 3px " + e.currentTarget.style.backgroundColor;
    maxz = (parseInt(e.currentTarget.style.zIndex + 1) > maxz) ? parseInt(e.currentTarget.style.zIndex) + 1 : maxz;
    e.currentTarget.style.zIndex = maxz;
    e.currentTarget.style.top = clamp(e.clientY, 50, document.getElementById('left').offsetHeight - e.currentTarget.clientHeight) + 'px';
    e.currentTarget.style.left = clamp(e.clientX, 10, document.getElementById('left').offsetWidth - e.currentTarget.clientWidth - 20) + 'px';
}

function dropBubble(e) {
    e.preventDefault();
    e.currentTarget.style.border = "none";
    e.currentTarget.style.boxShadow = "none";
    e.currentTarget.style.top = clamp(e.clientY, 50, document.getElementById('left').offsetHeight - e.currentTarget.clientHeight) + 'px';
    e.currentTarget.style.left = clamp(e.clientX, 10, document.getElementById('left').offsetWidth - e.currentTarget.clientWidth - 20) + 'px';
}

function clickBubble(e) {
	var names = {"Gender":"gender", "Age Category":"age_category", "Martial Status":"marital", "Monthly Income":"monthlyIncome", "Salary Hike":"salaryHike", "Perf Rating":"perf_rating", "Recognition":"recognition"};
    e.preventDefault();
    document.getElementById(names[e.currentTarget.innerText]).style = "display:block";
//    var submit = document.getElementById('submit');
//    var input = document.createElement("input");
//    input.type = "text";
//    input.placeholder = e.currentTarget.innerText;
//    input.setAttribute("th:field","*{"+ names[e.currentTarget.innerText]+"}")
//    $(input).insertBefore(submit);
   // }
}

function clamp(num, min, max) {
    return num <= min ? min : num >= max ? max : num;
}