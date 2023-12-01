"use client";
import React, { useState } from 'react'


import {Resizable} from 're-resizable';
import Draggable from 'react-draggable';

import Card from './card';
import multer from 'multer';
import path from 'path';

const imageUrl =
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRe7Ki-ys2G_MMb_xCrY7nAf87F5ZiIOyCh4f5H_JCTTtMSMLCL';
type listProps = {
    
};

const list:React.FC<listProps> = () => {
    const [cards, setCards] = useState([{url: "https://cdn.mos.cms.futurecdn.net/dPo92zYeAz7Joxh7HWooJ3-1200-80.jpg.webp"}]);
    
    const [files, setFiles] = useState(null);

    const handleFileChange = (e) => {
        setFiles(e.target.files);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!files || files.length === 0) {
            console.error('No files selected');
            return;
        }

        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('files[]', files[i]);
        }

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
            console.log(data);
        } catch (error) {
            console.error('Error uploading files:', error);
        }
    };

    return (
        <div>
            <h1>Upload Multiple Files</h1>
            <form onSubmit={handleSubmit} encType="multipart/form-data">
                <input type="file" name="files" multiple onChange={handleFileChange} />
                <button type="submit">Upload Files</button>
            </form>
        </div>
    );
}
export default list;