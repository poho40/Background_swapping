"use client";
// import React, { useState, useEffect } from 'react'
// import useDragger from "../hooks/useDragger";

// import {Resizable} from 're-resizable';
// import Draggable from 'react-draggable';

type cardProps = {
    url: string
};

// const card:React.FC<cardProps> = ({ url }) => {
//     const [x, setX]= useState(0)
//     const [y, setY]= useState(0)
//     const [w, setW]= useState(100)
//     const [h, setH]= useState(100)

//     useEffect(() => {
//         console.log('Current position X:', x);
//         console.log('Current position Y:', y);
//     }, [x, y]);

//     useEffect(() => {
//         console.log('Current width:', w);
//         console.log('Current height:', h);
//     }, [w, h]);

//     function handleStop(e: any, dragElement: any) {
//         setX(dragElement.x);
//         setY(dragElement.y);
//     }

//     function handleResizeStop(e: any, direction: any, ref: any, d: any) {
//         setW(ref.clientWidth);
//         setH(ref.clientHeight);
//     }

//     useDragger("mycard");


//     return(
//         // <Draggable axis="both"
//         // defaultPosition={{x: 0, y: 0}}
//         // position={{x: x, y:y}}
//         // onStop={handleStop} 
//         // grid={[25, 25]}
//         // bounds='parent'
//         // scale={1}
//         // >
//         //     <Resizable
//         //         defaultSize={{
//         //             width: 50,
//         //             height: 50,
//         //         }}
//         //         style={{
//         //             background: `url(${url})`,
//         //             backgroundSize: 'contain',
//         //             backgroundRepeat: 'no-repeat'
//         //         }}
//         //         onResizeStop={handleResizeStop}
//         //         bounds="parent"
//         //         >
//         //     </Resizable>
//         // </Draggable>

//         <div id="mycard" className="circle" ></div>
        
//     )
// }
// export default card;

import React from "react";
import useDragger from "../hooks/useDragger";

const Card: React.FC<cardProps> = ({ url })=> {
  
  useDragger("circle");

  return <div id="circle" className="circle"></div>
};

export default Card;