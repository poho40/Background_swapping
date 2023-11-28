"use client";
import React, { useState } from 'react'


import {Resizable} from 're-resizable';
import Draggable from 'react-draggable';

import Card from './card';

const imageUrl =
  'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRe7Ki-ys2G_MMb_xCrY7nAf87F5ZiIOyCh4f5H_JCTTtMSMLCL';
type listProps = {
    
};

const list:React.FC<listProps> = () => {
    const [cards, setCards] = useState([{url: "https://cdn.mos.cms.futurecdn.net/dPo92zYeAz7Joxh7HWooJ3-1200-80.jpg.webp"}]);
    
    
    return(
        <div className='parent pt-5 pl-5'>
            {cards.map((card) => <Card url={card.url} />)}
        </div>
    )
}
export default list;