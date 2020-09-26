var siteUrl = "https://localhost:5001/api"

function init() {

}

function findAll() {
  $.ajax({
    url: siteUrl + "/Stagione",
    type: "GET",
    contentType: "application/json",
    success: function (result) {
      readResult(JSON.stringify(result));
    },
    error: function (xhr, status, p3, p4) {
      var err = "Error " + " " + status + " " + p3;
      if (xhr.responseText && xhr.responseText[0] == "{")
        err = JSON.parse(xhr.responseText).message;
      alert(err);
    }
  });
}

function findById() {
  var id = $('#txtId').val();
  $.ajax({
    url: siteUrl + "/Stagione/" + id,
    type: "GET",
    contentType: "application/json",data: "",
    success: function (result) {
      readResult(JSON.stringify(result));
    },
    error: function (xhr, status, p3, p4) {
      var err = "Error " + " " + status + " " + p3;
      if (xhr.responseText && xhr.responseText[0] == "{")
        err = JSON.parse(xhr.responseText).message;
      alert(err);
    }
  });
}

function postItem() {
  var id    = $('#txtId').val();
  var anno = $('#txtNewAnno').val();
  var options = {};
  options.url = siteUrl + "/Stagione";
  options.type = "POST";
  options.data = JSON.stringify({"id": Number(id),"anno": Number(anno),"serie": 'C'});
  options.dataType = "json";
  options.contentType = "application/json";
  options.success = function (msg) { readResult(JSON.stringify(msg)); };
  options.error = function (err) { alert(err.responseText); };
  $.ajax(options);
}

function deleteId() {
  var options = {};
  // options.url = siteUrl + "/Clienti/deleteCustomer/"+ $("#txtId").val();
  options.url = siteUrl + "/Stagione/"+ $("#txtId").val();
  options.type = "DELETE";
  options.contentType = "application/json";
  options.success = function (msg) { readResult(JSON.stringify(msg)); };
  options.error = function (err) { alert(err.statusText); };
  $.ajax(options);
}

function updateId() {
  var id    = $('#txtId').val();
  var anno = $('#txtNewAnno').val();
  var options = {};
  options.url = siteUrl + "/Stagione/"+ $("#txtId").val();
  options.type = "PUT";
  options.data = JSON.stringify({"id": Number(id),"anno": Number(anno),"serie": 'C'});
  options.dataType = "json";
  options.contentType = "application/json";
  options.success = function (msg) { readResult(msg); };
  options.error   = function (err) { alert(err.responseText); };
  $.ajax(options);
}

function readResult(message) {
  console.log(message);
  // alert(message);
  document.querySelector('#risultatoChiamata').innerHTML = message;
  document.querySelector('#txtarea').text = message;
}
