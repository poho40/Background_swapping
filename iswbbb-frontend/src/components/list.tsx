"use client";
import React, { useState } from 'react'
import { useRouter } from 'next/navigation';


const imageUrl =
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRe7Ki-ys2G_MMb_xCrY7nAf87F5ZiIOyCh4f5H_JCTTtMSMLCL';
type listProps = {
    
};

const list:React.FC<listProps> = () => {    
    // const [files, setFiles] = useState([]);
    const [text, setText] = useState('');
    const [image, setImage] = useState('');
    const [back, setBack] = useState('');
    const router = useRouter();

    const handleTextChange = (e) => {
        setText(e.target.value);
    }

    const handleImageChange = (e) => {
        setImage(e.target.files[0]);
      };
    
      const handleBackChange = (e) => {
        setBack(e.target.files[0]);
      };
    // const handleFileChange = (e) => {
    //     setFiles(e.target.files);
    // };

    const handleSubmit = async (e) => {
        e.preventDefault();

        // if (!files || files.length === 0) {
        //     console.error('No files selected');
        //     return;
        // }

        const formData = new FormData();
        // for (let i = 0; i < files.length; i++) {
        //     formData.append('files[]', files[i]);
        // }
        formData.append('image', image)
        formData.append('back', back)
        formData.append('text', text)
        console.log(text)
        try {
            let headers = new Headers();
            headers.append('Content-Type','application/json')
            headers.append('Access-Control-Allow-Origin', 'http://127.0.0.1:8080/')
            const response = await fetch('http://127.0.0.1:8080/upload-multiple', {
                method: 'POST',
                body: formData,
            });

            // Handle the response as needed
            const data = await response.json();
            router.push('/background');
        } catch (error) {
            console.error('Error uploading files:', error);
        }
    };

    return (
        <div>
            
            <form onSubmit={handleSubmit} encType="multipart/form-data">
                <div>
                <input onChange={handleTextChange} type="text" id="first_name" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-1/2 p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Enter Generated Background" required></input>
                </div>
                <div>Select Image
                <input type="file" name="files"  accept=".png" onChange={handleImageChange} />
                </div>
                <div>Select Back
                <input type="file" name="files" accept=".png" onChange={handleBackChange} />
                </div>
                <button type="submit">Submit</button>
            </form>
        </div>
    );
}
export default list;