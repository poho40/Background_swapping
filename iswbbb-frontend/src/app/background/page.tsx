"use client";
import React, { useState } from 'react'


import {Resizable} from 're-resizable';
import Draggable from 'react-draggable';

import Card from '@/components/card';

const imageUrl =
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRe7Ki-ys2G_MMb_xCrY7nAf87F5ZiIOyCh4f5H_JCTTtMSMLCL';
type listProps = {
    
};

const background:React.FC<listProps> = () => {
  const [cards, setCards] = useState([{url: "https://www.comingsoon.net/wp-content/uploads/sites/3/2023/06/Watch-the-Transformers-Movies-Before-Rise-of-the-Beasts.jpg"},{url: "/background/transformers.png"}]);
  console.log(cards)
    return(
        <div className='parent pt-5 pl-5'>
            {cards.map((card) => <Card url= {card.url} />)}
        </div>
    )
}
export default background;