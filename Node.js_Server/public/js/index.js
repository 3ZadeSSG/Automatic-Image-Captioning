var original_img_loc = null;

/*
Uploads the selected file to Node.js server
Keeps track of progress and shows on progress bar
*/
function uploadFiles(formData) {
  $.ajax({
      url: '/upload_photos',
      method: 'post',
      data: formData,
      processData: false,
      contentType: false,
      xhr: function () {
          var xhr = new XMLHttpRequest();
          xhr.upload.addEventListener('progress', function (event) {
              var progressBar = $('#upload_progress_bar');
              if (event.lengthComputable) {
                  var percent = (event.loaded / event.total) * 100;
                  progressBar.width(percent + '%');
              }
          });
          return xhr;
      }
  }).done(handleSuccess).fail(function (xhr, status) {
    $('#button_captions').css('visibility','hidden');
      alert(status);
  });
}

/*
If image uplad is successful then show the image on view and make 'generate captions' button visible
*/
function handleSuccess(data) {
  if (data.length > 0) {
      var html = '';
      let initImg = false;
      for (var i=0; i < data.length; i++) {
          var img = data[i];

          if (img.status) {
              html += '<div class="col-xs-6 col-md-4" style="text-align:center;"><img src="' + img.publicPath + '" alt="' + img.filename  + '"style="max-width: 400px; max-height: 450px; height: auto; width: auto;"></div>';
              $('#button_captions').css('visibility','visible');
              original_img_loc = img.publicPath;
          } else {
              html += '<div class="col-xs-6 col-md-4"><a href="#" class="thumbnail">Invalid file type - ' + img.filename  + '</a></div>';
          }
      }
      $('#album').html(html);

  } else {
      alert('No images were uploaded.')
  }
}

$('#photos-input').on('change', function () {
  // if new file is selected then reset and hide progress bar (will be made visible when upload button is clicked)
  $('#upload_progress_bar').width('0%');
  $('#progress_upload').css('visibility','hidden');
});


$('#upload-photos').on('submit', function (event) {
    $('#progress_upload').css('visibility','visible');
    original_img_loc = null;

    event.preventDefault();

    var files = $('#photos-input').get(0).files,
        formData = new FormData();

  if (files.length === 0) {
      alert('Select atleast 1 file to upload.');
      return false;
  }

  if (files.length > 1) {
      alert('You can only upload up to 1 file.');
      return false;
  }

  // Keeping this loop since in case multiple files support is added then only above if condition needs to be changed
  for (var i=0; i < files.length; i++) {
      var file = files[i];
      formData.append('photos[]', file, file.name);
  }
  uploadFiles(formData);
});


$(document).ready(function() {
    $('#button_captions').on('click',()=>{
        requestcaptions(original_img_loc);
    });
});


/*
* Requests server  to return caption for the image given
*/
function requestcaptions(image_path){
    $('#progress_captions').css('visibility','visible');
    $("#output_caption").empty();
    $.post({
        traditional: true,
        url: '/predictCaption',
        contentType: 'application/json',
        dataType: 'json',
        data: JSON.stringify({'filePath' : image_path}),
        success:function(response) {
            $('#progress_captions').css('visibility','hidden');
            renderResults(response.result);
        },
        error: function(response, textStatus, error){
            $('#progress_captions').css('visibility','hidden');
            alert(error);
        }
    });
}

/*
Appends caption text into page
*/
function renderResults(data){
    $('#hint_bottom').css('visibility','visible');
    $('#button_captions').css('visibility','hidden');
    console.log(data);
    $("#output_caption").append('<div style="max-width: 400px;display: block;margin: auto;">\
                                    <h5 class="header">Predicted Caption</h5>\
                                    <div class="card horizontal">\
                                        <div class="card-stacked">\
                                            <div class="card-content">\
                                                <p id="text_predicted_caption">'+data.sentence+'</p>\
                                            </div>\
                                        </div>\
                                    </div>\
                                </div>');
}