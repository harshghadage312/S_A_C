function showUpload() {
  document.getElementById("upload-section").classList.remove("hidden");
}

function openCamera() {
  alert("📷 Camera feature will be integrated soon!");
}

function previewImage(input, previewId) {
  const file = input.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const img = document.getElementById(previewId);
      img.src = e.target.result;
      img.style.display = "block";
    }
    reader.readAsDataURL(file);
  }
}