import React, { useState } from "react";
import './Home.css';
import axios from "axios";

const InputField = ({ label, value, onChange }) => (
  <div className="input-field">
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      required
    />
    <label>{label}</label>
  </div>
);

const SelectField = ({ label, options, value, onChange }) => (
  <div className="select-field">
    <select value={value} onChange={(e) => onChange(e.target.value)} required>
      <option value="" disabled>
        
      </option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
    <label>{label}</label>
  </div>
);

const Home = () => {
  const [formData, setFormData] = useState({
    height_of_video_wall: "",
    width_of_video_wall: "",
    room_length: "",
    room_width: "",
    room_height: "",
    seating_type: "",
    number_of_seats: "",
  });

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log(formData);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData
      );

      if (response.data) {
        window.alert("Configurations sent!");
      } else {
        window.alert("Something went wrong. Please try again.");
      }
    } catch (err) {
      console.log(err);
      window.alert("Error sending data. Please check the console for details.");
    }
  };

  return (
    <div className="home-container">
      <h2>Video Wall Configuration</h2>

      <form onSubmit={handleSubmit} className="form-container">
        <div className="form-grid">
          <InputField
            label="Height of Video Wall"
            value={formData.height_of_video_wall}
            onChange={(value) =>
              setFormData({ ...formData, height_of_video_wall: value })
            }
          />
          <InputField
            label="Width of Video Wall"
            value={formData.width_of_video_wall}
            onChange={(value) =>
              setFormData({ ...formData, width_of_video_wall: value })
            }
          />
          <InputField
            label="Room Length"
            value={formData.room_length}
            onChange={(value) =>
              setFormData({ ...formData, room_length: value })
            }
          />
          <InputField
            label="Room Width"
            value={formData.room_width}
            onChange={(value) =>
              setFormData({ ...formData, room_width: value })
            }
          />
          <InputField
            label="Room Height"
            value={formData.room_height}
            onChange={(value) =>
              setFormData({ ...formData, room_height: value })
            }
          />
          <SelectField
            label="Seating Type"
            options={["cluster", "row"]}
            value={formData.seating_type}
            onChange={(value) =>
              setFormData({ ...formData, seating_type: value })
            }
          />
          <InputField
            label="Number of Seats"
            value={formData.number_of_seats}
            onChange={(value) =>
              setFormData({ ...formData, number_of_seats: value })
            }
          />
        </div>
        <button type="submit" className="submit-btn">
          Submit
        </button>
      </form>
    </div>
  );
};

export default Home;
