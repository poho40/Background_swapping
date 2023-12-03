"use client";
import React, { useState, useEffect } from 'react'
import { useImageSize } from 'react-image-size';
import Card from '@/components/Card'
import '../../App.css'

const imageUrl =
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRe7Ki-ys2G_MMb_xCrY7nAf87F5ZiIOyCh4f5H_JCTTtMSMLCL';
type listProps = {
    
};

const background:React.FC<listProps> = () => {
  const [cards, setCards] = useState([]);
  // useEffect(() => {
  //   // Using fetch to fetch the api from 
  //   // flask server it will be redirected to proxy
  //   fetch("http://localhost:8080/getnames").then((res) =>
  //       res.json().then((data) => {
  //           // Setting a data from api
  //           let temp:any = []
  //           for (let i = 0; i < data.files.length; ++i){
  //             let str = "/background/" + data.files[i]
  //             temp.push({url:str, dimensions: {x:0, y:0, width:200, height:200}})
  //           }
  //           setCards(temp)
  //       })
  //   );
  // }, []);
  // const handleCardUpdate = (url:string, x: number, y: number, w: number, h: number) => {
  //   // Handle the updated values here, if needed
  //   let updatedCards = [...cards]
  //   for (let i = 0;i < updatedCards.length; ++i) {
  //     if (updatedCards[i]['url'] == url) {
  //       updatedCards[i]['dimensions'].x = x
  //       updatedCards[i]['dimensions'].y = y
  //       updatedCards[i]['dimensions'].width = w
  //       updatedCards[i]['dimensions'].height = h
  //     }
  //   }
  //   setCards(updatedCards)
  //   console.log('Updated values received from Card:', url, x, y, w, h);
  // };

  // const handleClick = async (e) => {
  //   const formData = new FormData();
  //   e.preventDefault();
  //   const requestBody = JSON.stringify({ cards });
  //   try {
  //       let headers = new Headers();
  //       headers.append('Content-Type','application/json')
  //       headers.append('Access-Control-Allow-Origin', 'http://127.0.0.1:8080/')
  //       const response = await fetch('http://127.0.0.1:8080/background', {
  //           method: 'POST',
  //           headers: headers,
  //           body:requestBody,
  //       });

  //       // Handle the response as needed
  //       const data = await response.json();
  //   } catch (error) {
  //       console.error('Error uploading files:', error);
  //   }
  // };
  const handleClick = () => {
    const imageUrl = '/background/442_compose.png'; // Replace with the actual image URL

    // Create a virtual anchor element
    const downloadLink = document.createElement('a');

    // Set the href and download attributes
    downloadLink.href = imageUrl;
    downloadLink.download = 'downloaded_image.png'; // Set the desired file name

    // Append the link to the document
    document.body.appendChild(downloadLink);

    // Trigger a click on the link to start the download
    downloadLink.click();

    // Remove the link from the document
    document.body.removeChild(downloadLink);
  };

    return(
        <div>
        <img src='/background/442_compose.png' width='1920' height='1080'/>
        <button onClick={handleClick}>Download</button>
        </div>
    )
}
export default background;