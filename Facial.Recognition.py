import streamlit as st
import cv2
import os
import numpy as np
from datetime import datetime

# Define the face cascade classifier - make sure the path is correct for Colab
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def hex_to_bgr(hex_color):
    """Convert hex color to BGR format for OpenCV"""
    # Remove the '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert hex to RGB
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Convert RGB to BGR for OpenCV
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr

def detect_faces_from_webcam(rect_color, min_neighbors, scale_factor, save_images):
    """Detect faces from webcam with customizable parameters"""
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Error: Could not open webcam. Please check your camera permissions.")
        return

    # Create placeholder for the video stream
    stframe = st.empty()
    
    # Create columns for controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stop_button = st.button("🛑 Stop Detection", key="stop_webcam")
    
    with col2:
        if save_images:
            capture_button = st.button("📸 Capture Frame", key="capture_webcam")
        else:
            capture_button = False
    
    with col3:
        faces_count = st.empty()

    frame_count = 0
    
    try:
        while True:
            # Read the frames from the webcam
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Error: Could not read frame from webcam.")
                break

            # Convert the frames to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the faces using the face cascade classifier with custom parameters
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )

            # Draw rectangles around the detected faces with custom color
            bgr_color = hex_to_bgr(rect_color)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)
                # Add face number label
                cv2.putText(frame, f'Face {len(faces)}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)

            # Update face count
            faces_count.metric("👤 Faces Detected", len(faces))

            # Display the frames in the Streamlit app
            stframe.image(frame, channels="BGR", use_column_width=True)

            # Save image if capture button is pressed
            if capture_button and save_images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"face_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                st.success(f"📸 Image saved as {filename}")

            # Check if stop button is pressed
            if stop_button:
                break
                
            frame_count += 1

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Release the webcam
        cap.release()
        st.info("📹 Webcam released successfully")

def detect_faces_from_image(uploaded_file, rect_color, min_neighbors, scale_factor, save_images):
    """Detect faces from uploaded image"""
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        
        # Draw rectangles around detected faces
        bgr_color = hex_to_bgr(rect_color)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), bgr_color, 2)
            cv2.putText(image, f'Face {len(faces)}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Detection Results")
            st.metric("👤 Faces Detected", len(faces))
            
            if save_images and len(faces) > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"face_detection_image_{timestamp}.jpg"
                cv2.imwrite(filename, image)
                st.success(f"📸 Processed image saved as {filename}")
        
        with col2:
            st.subheader("🖼️ Processed Image")
            st.image(image, channels="BGR", use_column_width=True)

def app():
    st.set_page_config(
        page_title="Face Detection App",
        page_icon="👤",
        layout="wide"
    )
    
    st.title("👤 Face Detection using Viola-Jones Algorithm")
    
    # Instructions
    st.markdown("""
    ### 📋 Instructions
    
    **Welcome to the Face Detection App!** This application uses the Viola-Jones algorithm to detect faces in real-time.
    
    **How to use:**
    1. 🎨 **Customize Detection**: Use the sidebar controls to adjust detection parameters
    2. 🎯 **Choose Input Source**: Select between webcam or image upload
    3. 📸 **Save Results**: Enable image saving to capture your detections
    4. 🔧 **Fine-tune**: Adjust the sliders to optimize detection for your use case
    
    **Parameter Guide:**
    - **Scale Factor**: How much the image size is reduced at each scale (1.1 = more detections but slower)
    - **Min Neighbors**: How many neighbors each face rectangle should have to retain it (higher = fewer false positives)
    - **Rectangle Color**: Choose your preferred color for the detection rectangles
    """)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Detection Controls")
    
    # Color picker for rectangle color
    rect_color = st.sidebar.color_picker(
        "🎨 Choose Rectangle Color", 
        "#00FF00",  # Default green
        help="Select the color for face detection rectangles"
    )
    
    # Slider for minNeighbors parameter
    min_neighbors = st.sidebar.slider(
        "👥 Min Neighbors",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Higher values result in fewer detections but with higher quality"
    )
    
    # Slider for scaleFactor parameter
    scale_factor = st.sidebar.slider(
        "🔍 Scale Factor",
        min_value=1.1,
        max_value=2.0,
        value=1.3,
        step=0.1,
        help="How much the image size is reduced at each scale"
    )
    
    # Checkbox for saving images
    save_images = st.sidebar.checkbox(
        "💾 Save Detection Images",
        value=False,
        help="Save processed images with face detections"
    )
    
    # Input source selection
    st.sidebar.header("📥 Input Source")
    input_source = st.sidebar.radio(
        "Choose input source:",
        ["📹 Webcam", "🖼️ Upload Image"],
        help="Select whether to use webcam or upload an image"
    )
    
    # Main content area
    if input_source == "📹 Webcam":
        st.header("📹 Real-time Face Detection")
        
        st.markdown("""
        **Webcam Instructions:**
        - Click "Start Detection" to begin real-time face detection
        - Use "Stop Detection" to end the session
        - If image saving is enabled, use "Capture Frame" to save the current frame
        - Make sure your browser allows camera access
        """)
        
        # Add a button to start detecting faces
        if st.button("🚀 Start Detection", type="primary"):
            detect_faces_from_webcam(rect_color, min_neighbors, scale_factor, save_images)
    
    else:  # Upload Image
        st.header("🖼️ Image Face Detection")
        
        st.markdown("""
        **Image Upload Instructions:**
        - Upload an image using the file uploader below
        - Supported formats: JPG, JPEG, PNG
        - The app will automatically detect faces and display results
        - Processed images can be saved if the option is enabled
        """)
        
        uploaded_file = st.file_uploader(
            "📤 Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to detect faces"
        )
        
        if uploaded_file is not None:
            detect_faces_from_image(uploaded_file, rect_color, min_neighbors, scale_factor, save_images)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **💡 Tips for Better Detection:**
    - Ensure good lighting conditions
    - Face the camera directly
    - Maintain a reasonable distance from the camera
    - Adjust parameters if detection is too sensitive or missing faces
    
    **🔧 Troubleshooting:**
    - If no faces are detected, try lowering the Min Neighbors value
    - If too many false positives, increase the Min Neighbors value
    - If detection is slow, increase the Scale Factor value
    """)

if __name__ == "__main__":
    app()