"use client";
import React, { useState, useEffect } from 'react'
import Circle from '@/components/Circle';
import PinkBox from '@/components/PinkBox';
import SmallCircle from '@/components/SmallCircle';
import Card from '@/components/Card'
import '../../App.css'

const imageUrl =
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRe7Ki-ys2G_MMb_xCrY7nAf87F5ZiIOyCh4f5H_JCTTtMSMLCL';
type listProps = {
    
};

const background:React.FC<listProps> = () => {
  const [cards, setCards] = useState([]);
  useEffect(() => {
    // Using fetch to fetch the api from 
    // flask server it will be redirected to proxy
    fetch("http://localhost:8080/getnames").then((res) =>
        res.json().then((data) => {
            // Setting a data from api
            let temp:any = []
            for (let i = 0; i < data.files.length; ++i){
              temp.push({url: "/background/" + data.files[i]})
            }
            setCards(temp)
        })
    );
  }, []);

  console.log(cards)

    return(
        <div>
        <div className="container">
        {cards.map((card) => <Card url= {card.url} />)}
        {/* <Card url='' /> */}
        {/* <Circle />
        <SmallCircle /> */}
        </div>
        </div>
    )
}
export default background;